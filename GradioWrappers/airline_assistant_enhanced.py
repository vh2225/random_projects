import gradio as gr
import ollama
import edge_tts
import asyncio
import tempfile
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import warnings
try:
    import speech_recognition as sr
except ImportError:
    sr = None
    print("Speech recognition not available")
from datetime import datetime
warnings.filterwarnings("ignore")

# Initialize models
MODEL = "gemma2:2b"
device = "cpu"
dtype = torch.float32

# Check Ollama
try:
    ollama.list()
    print("✅ Ollama is running!")
except:
    print("❌ Ollama is not running. Please start it with 'ollama serve'")

# Ticket prices
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

# TTS voices for different languages
VOICES = {
    "en": {
        "Female Assistant": "en-US-AriaNeural",
        "Male Assistant": "en-US-GuyNeural",
        "Friendly Female": "en-US-JennyNeural",
    },
    "es": {
        "Female Assistant": "es-ES-ElviraNeural",
        "Male Assistant": "es-ES-AlvaroNeural",
    },
    "zh": {
        "Female Assistant": "zh-CN-XiaoxiaoNeural",
        "Male Assistant": "zh-CN-YunxiNeural",
    },
    "de": {
        "Female Assistant": "de-DE-KatjaNeural",
        "Male Assistant": "de-DE-ConradNeural",
    }
}

# Translation prompts
TRANSLATION_PROMPTS = {
    "es": "Translate the following English text to Spanish. Only return the translation, nothing else: ",
    "zh": "Translate the following English text to Chinese. Only return the translation, nothing else: ",
    "de": "Translate the following English text to German. Only return the translation, nothing else: "
}

# Global variables
sd_pipe = None
booking_state = {"can_book": False, "city": None, "price": None}

def load_sd_model():
    """Load Stable Diffusion model"""
    global sd_pipe
    if sd_pipe is None:
        print("Loading Stable Diffusion model...")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        )
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.enable_attention_slicing()
        print("✅ Image model loaded!")
    return sd_pipe

# Enhanced system message with prices
price_info = "Current return ticket prices:\n"
for city, price in ticket_prices.items():
    price_info += f"- {city.capitalize()}: {price}\n"

system_message = f"""You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 2 sentences.
Always be accurate. If you don't know the answer, say so.

{price_info}

When customers ask about ticket prices, provide the exact price from the list above.
If they ask about a destination, mention something interesting about that city.
When you tell them a price, always ask if they would like to book the flight."""

async def generate_speech(text, voice_name, language="en"):
    """Generate speech from text"""
    try:
        voices = VOICES.get(language, VOICES["en"])
        voice = voices.get(voice_name, list(voices.values())[0])
        communicate = edge_tts.Communicate(text, voice)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
        
        await communicate.save(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def translate_text(text, target_language):
    """Translate text using Ollama"""
    if target_language == "en":
        return text
    
    prompt = TRANSLATION_PROMPTS.get(target_language, "")
    if not prompt:
        return text
    
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt + text}],
        )
        return response['message']['content']
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_city_image(city):
    """Generate image of a city"""
    try:
        pipeline = load_sd_model()
        prompt = f"Beautiful tourist destination {city}, travel poster style, vibrant colors, landmarks"
        
        print(f"Generating image for {city}...")
        image = pipeline(
            prompt=prompt,
            negative_prompt="low quality, blurry",
            num_inference_steps=20,
            height=384,
            width=384
        ).images[0]
        
        # Upscale for better display
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def process_voice_input(audio):
    """Convert speech to text"""
    if audio is None:
        return ""
    
    if sr is None:
        return "Speech recognition not available"
    
    try:
        recognizer = sr.Recognizer()
        # Convert audio to wav format that speech_recognition can handle
        audio_data = sr.AudioFile(audio)
        with audio_data as source:
            audio_data = recognizer.record(source)
        
        # Try multiple languages
        text = ""
        languages = ["en-US", "es-ES", "zh-CN", "de-DE"]
        for lang in languages:
            try:
                text = recognizer.recognize_google(audio_data, language=lang)
                if text:
                    break
            except:
                continue
        
        if not text:
            return "Could not understand audio"
        
        return text
    except ImportError:
        return "Voice input not available. Please install pyaudio."
    except Exception as e:
        print(f"Voice recognition error: {e}")
        return "Could not process audio"

def check_booking_eligibility(message, reply):
    """Check if booking should be enabled"""
    global booking_state
    
    # Reset booking state
    booking_state = {"can_book": False, "city": None, "price": None}
    
    # Check if reply contains price and booking question
    message_lower = message.lower()
    reply_lower = reply.lower()
    
    for city, price in ticket_prices.items():
        if (city in message_lower or city in reply_lower) and price in reply:
            if "book" in reply_lower or "would you like" in reply_lower:
                booking_state = {"can_book": True, "city": city.capitalize(), "price": price}
                return True
    
    return False

def handle_booking():
    """Handle flight booking"""
    if not booking_state["can_book"]:
        return None
    
    # Log booking to console
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    booking_info = f"BOOKING CONFIRMED - {timestamp} - Destination: {booking_state['city']} - Price: {booking_state['price']}"
    print(f"\n{'='*50}")
    print(booking_info)
    print(f"{'='*50}\n")
    
    # Create confirmation message
    confirmation = f"✅ Flight booked! Your flight to {booking_state['city']} has been confirmed. Total price: {booking_state['price']}. You will receive a confirmation email shortly."
    
    # Reset booking state
    booking_state["can_book"] = False
    
    return confirmation

def chat_with_assistant(message, history, voice_choice, generate_images, target_language):
    """Main chat function"""
    
    # Build messages for Ollama
    messages = [{"role": "system", "content": system_message}]
    
    # Add history
    for msg in history:
        if msg['role'] in ['user', 'assistant']:
            messages.append({"role": msg['role'], "content": msg['content']})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Get response from Ollama
    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
        )
        reply = response['message']['content']
    except Exception as e:
        reply = f"I apologize, but I'm having trouble connecting to the language model. Error: {str(e)}"
    
    # Check if booking should be enabled
    can_book = check_booking_eligibility(message, reply)
    
    # Translate reply if needed
    translated_reply = translate_text(reply, target_language)
    
    # Generate speech for both languages
    audio_path_en = None
    audio_path_translated = None
    if voice_choice != "No Voice":
        audio_path_en = asyncio.run(generate_speech(reply, voice_choice, "en"))
        if target_language != "en":
            audio_path_translated = asyncio.run(generate_speech(translated_reply, voice_choice, target_language))
    
    # Check if we should generate an image
    image = None
    if generate_images:
        # Simple keyword detection for cities
        city_keywords = {
            "london": ["london", "big ben", "thames"],
            "paris": ["paris", "eiffel", "louvre"],
            "tokyo": ["tokyo", "japan", "shibuya"],
            "berlin": ["berlin", "brandenburg", "germany"]
        }
        
        for city, keywords in city_keywords.items():
            if any(keyword in message.lower() for keyword in keywords):
                image = generate_city_image(city.capitalize())
                break
    
    return reply, translated_reply, audio_path_en, audio_path_translated, image, can_book

# Create Gradio interface
with gr.Blocks(title="✈️ FlightAI Assistant Enhanced") as demo:
    gr.Markdown("""
    # ✈️ FlightAI Assistant - Enhanced Edition
    
    Chat with our AI flight assistant! Now with multilingual support and voice input.
    
    **New Features:**
    - 🎤 Voice input support
    - 🌍 Multilingual responses (Spanish, Chinese, German)
    - 📝 Flight booking capability
    - 🎙️ Multilingual text-to-speech
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=400,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Ask about flights, prices, or destinations:",
                    placeholder="e.g., How much is a ticket to Paris?",
                    lines=2,
                    scale=4
                )
                
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤",
                    scale=1,
                    show_label=False
                )
            
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    choices=["No Voice"] + list(VOICES["en"].keys()),
                    value="Female Assistant",
                    label="Voice"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=[("English", "en"), ("Spanish", "es"), ("Chinese", "zh"), ("German", "de")],
                    value="en",
                    label="Translation Language"
                )
                
                image_check = gr.Checkbox(
                    label="Generate destination images (slower)",
                    value=False
                )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                book_btn = gr.Button("Book Flight", variant="secondary", interactive=False)
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            # Split panel for responses
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### English Response")
                    english_text = gr.Textbox(label="", lines=4, interactive=False)
                    audio_output_en = gr.Audio(
                        label="English Voice",
                        type="filepath",
                        autoplay=True
                    )
                
                with gr.Column():
                    gr.Markdown("### Translated Response")
                    translated_text = gr.Textbox(label="", lines=4, interactive=False)
                    audio_output_translated = gr.Audio(
                        label="Translated Voice",
                        type="filepath",
                        autoplay=False
                    )
            
            image_output = gr.Image(
                label="Destination",
                type="pil"
            )
    
    # Examples
    gr.Examples(
        examples=[
            "How much is a ticket to Paris?",
            "Tell me about Tokyo",
            "What's the price for London?",
            "I want to visit Berlin",
            "Which destination is cheapest?",
        ],
        inputs=msg
    )
    
    def respond(message, history, voice, gen_images, lang, voice_audio):
        # Process voice input if provided
        if voice_audio:
            message = process_voice_input(voice_audio)
            if not message:
                message = "Could not understand audio"
        
        reply, translated, audio_en, audio_trans, image, can_book = chat_with_assistant(
            message, history, voice, gen_images, lang
        )
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        
        return (
            history, 
            reply, 
            translated, 
            audio_en, 
            audio_trans, 
            image, 
            "", 
            None,  # Clear voice input
            gr.update(interactive=can_book)  # Update book button state
        )
    
    def book_flight(history):
        confirmation = handle_booking()
        if confirmation:
            history.append({"role": "assistant", "content": confirmation})
            return history, gr.update(interactive=False)
        return history, gr.update(interactive=False)
    
    # Event handlers
    msg.submit(
        respond,
        [msg, chatbot, voice_dropdown, image_check, language_dropdown, voice_input],
        [chatbot, english_text, translated_text, audio_output_en, audio_output_translated, 
         image_output, msg, voice_input, book_btn]
    )
    
    submit.click(
        respond,
        [msg, chatbot, voice_dropdown, image_check, language_dropdown, voice_input],
        [chatbot, english_text, translated_text, audio_output_en, audio_output_translated, 
         image_output, msg, voice_input, book_btn]
    )
    
    book_btn.click(
        book_flight,
        [chatbot],
        [chatbot, book_btn]
    )
    
    # Process voice input when recorded
    voice_input.change(
        lambda x: process_voice_input(x) if x else "",
        [voice_input],
        [msg]
    )
    
    clear.click(
        lambda: ([], "", "", None, None, None, gr.update(interactive=False)), 
        None, 
        [chatbot, english_text, translated_text, audio_output_en, 
         audio_output_translated, image_output, book_btn]
    )
    
    gr.Markdown("""
    ---
    **Notes:**
    - Voice input supports English, Spanish, Chinese, and German (requires pyaudio)
    - Book button activates when the assistant offers to book a flight
    - First image generation will download the model (~5GB)
    - Make sure Ollama is running (`ollama serve`)
    - TTS requires internet connection
    """)

if __name__ == "__main__":
    print("\n🚀 Starting Enhanced FlightAI Assistant...")
    
    # Check and install required packages
    required_packages = ["edge-tts", "SpeechRecognition", "pydub"]
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            try:
                subprocess.check_call(["pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install manually.")
    
    # Check for pyaudio separately (optional for voice input)
    try:
        import pyaudio
        print("✅ Voice input ready")
    except ImportError:
        print("⚠️  PyAudio not installed. Voice input may not work.")
        print("   To enable voice input on macOS:")
        print("   1. Install PortAudio: brew install portaudio")
        print("   2. Install PyAudio: pip install pyaudio")
        print("   Continuing without voice input support...")
    
    # Check for Ollama model
    try:
        ollama.show(MODEL)
        print(f"✅ {MODEL} is ready")
    except:
        print(f"📥 Downloading {MODEL}...")
        ollama.pull(MODEL)
    
    print("\n📱 Opening in your browser...\n")
    demo.launch(share=True)