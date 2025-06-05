import gradio as gr
import ollama
import edge_tts
import asyncio
import tempfile
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import warnings
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

# TTS voices
VOICES = {
    "Female Assistant": "en-US-AriaNeural",
    "Male Assistant": "en-US-GuyNeural",
    "Friendly Female": "en-US-JennyNeural",
}

# Global variables for models
sd_pipe = None

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
If they ask about a destination, mention something interesting about that city."""

async def generate_speech(text, voice_name):
    """Generate speech from text"""
    try:
        voice = VOICES[voice_name]
        communicate = edge_tts.Communicate(text, voice)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
        
        await communicate.save(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

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

def chat_with_assistant(message, history, voice_choice, generate_images):
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
    
    # Generate speech
    audio_path = None
    if voice_choice != "No Voice":
        audio_path = asyncio.run(generate_speech(reply, voice_choice))
    
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
    crea
    return reply, audio_path, image

# Create Gradio interface
with gr.Blocks(title="✈️ FlightAI Assistant") as demo:
    gr.Markdown("""
    # ✈️ FlightAI Assistant - Free & Open Source
    
    Chat with our AI flight assistant! Powered by local models - no API keys needed.
    
    **Features:**
    - 💬 Chat with Ollama (local LLM)
    - 🎙️ Text-to-speech responses
    - 🎨 Destination images (when enabled)
    - 💰 Current ticket prices for London, Paris, Tokyo, and Berlin
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=400,
                type="messages"
            )
            
            msg = gr.Textbox(
                label="Ask about flights, prices, or destinations:",
                placeholder="e.g., How much is a ticket to Paris?",
                lines=2
            )
            
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    choices=["No Voice"] + list(VOICES.keys()),
                    value="Female Assistant",
                    label="Voice"
                )
                
                image_check = gr.Checkbox(
                    label="Generate destination images (slower)",
                    value=False
                )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Assistant Voice",
                type="filepath",
                autoplay=True
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
    
    def respond(message, history, voice, gen_images):
        reply, audio, image = chat_with_assistant(message, history, voice, gen_images)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, audio, image, ""
    
    # Event handlers
    msg.submit(
        respond,
        [msg, chatbot, voice_dropdown, image_check],
        [chatbot, audio_output, image_output, msg]
    )
    
    submit.click(
        respond,
        [msg, chatbot, voice_dropdown, image_check],
        [chatbot, audio_output, image_output, msg]
    )
    
    clear.click(lambda: ([], None, None), None, [chatbot, audio_output, image_output])
    
    gr.Markdown("""
    ---
    **Notes:**
    - First image generation will download the model (~5GB)
    - Image generation is CPU-based and takes 30-60 seconds
    - Make sure Ollama is running (`ollama serve`)
    - TTS requires internet connection
    """)

if __name__ == "__main__":
    print("\n🚀 Starting FlightAI Assistant...")
    
    # Check and install edge-tts if needed
    try:
        import edge_tts
    except ImportError:
        print("Installing edge-tts...")
        import subprocess
        subprocess.check_call(["pip", "install", "edge-tts"])
    
    # Check for Ollama model
    try:
        ollama.show(MODEL)
        print(f"✅ {MODEL} is ready")
    except:
        print(f"📥 Downloading {MODEL}...")
        ollama.pull(MODEL)
    
    print("\n📱 Opening in your browser...\n")
    demo.launch(share=True)