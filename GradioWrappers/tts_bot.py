import gradio as gr
import edge_tts
import asyncio
import tempfile
import os

# Available voices
VOICES = {
    "Female - US": "en-US-AriaNeural",
    "Male - US": "en-US-GuyNeural", 
    "Female - UK": "en-GB-SoniaNeural",
    "Male - UK": "en-GB-RyanNeural",
    "Female - Friendly": "en-US-JennyNeural",
    "Male - Newscast": "en-US-ChristopherNeural",
}

async def text_to_speech(text, voice_name, rate, pitch):
    """Convert text to speech using Edge-TTS"""
    
    if not text:
        return None, "Please enter some text"
    
    try:
        # Get voice ID
        voice = VOICES[voice_name]
        
        # Adjust rate and pitch
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        
        # Create communication object
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate_str,
            pitch=pitch_str
        )
        
        # Generate speech to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
            
        await communicate.save(tmp_path)
        
        return tmp_path, f"✅ Generated speech using {voice_name}"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def generate_speech(text, voice, rate, pitch):
    """Wrapper to run async function"""
    return asyncio.run(text_to_speech(text, voice, rate, pitch))

# Create Gradio interface
with gr.Blocks(title="🎙️ Free Text-to-Speech") as demo:
    gr.Markdown("""
    # 🎙️ Free Text-to-Speech Generator
    
    Convert text to natural-sounding speech using Microsoft Edge's TTS engine.
    
    **Features:**
    - Multiple voices and accents
    - Adjustable speed and pitch
    - Completely free - no API keys needed!
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Type or paste your text here...",
                lines=6
            )
            
            voice_dropdown = gr.Dropdown(
                choices=list(VOICES.keys()),
                value="Female - US",
                label="Voice"
            )
            
            with gr.Row():
                rate_slider = gr.Slider(
                    minimum=-50,
                    maximum=50,
                    value=0,
                    step=10,
                    label="Speed adjustment (%)"
                )
                
                pitch_slider = gr.Slider(
                    minimum=-50,
                    maximum=50,
                    value=0,
                    step=10,
                    label="Pitch adjustment (Hz)"
                )
            
            generate_btn = gr.Button("🎙️ Generate Speech", variant="primary")
            
            status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath"
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Welcome to FlightAI! How may I assist you today?", "Female - Friendly", 0, 0],
            ["Your flight to Paris departs at 3:30 PM from gate B12.", "Male - US", -10, 0],
            ["Breaking news: Scientists discover new planet!", "Male - Newscast", 10, -10],
            ["Once upon a time, in a land far away...", "Female - UK", -20, 20],
        ],
        inputs=[text_input, voice_dropdown, rate_slider, pitch_slider],
    )
    
    # Event handlers
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, voice_dropdown, rate_slider, pitch_slider],
        outputs=[audio_output, status]
    )
    
    gr.Markdown("""
    ---
    **Tips:**
    - **Speed**: Negative = slower, Positive = faster
    - **Pitch**: Negative = lower voice, Positive = higher voice
    - Try different voices to find the perfect one for your use case
    
    **Note:** First generation might take a moment to initialize.
    """)

if __name__ == "__main__":
    print("🎙️ Starting Text-to-Speech Generator...")
    print("📱 Opening in your browser...\n")
    
    # Install edge-tts if not installed
    try:
        import edge_tts
    except ImportError:
        print("Installing edge-tts...")
        import subprocess
        subprocess.check_call(["pip", "install", "edge-tts"])
        print("✅ edge-tts installed!\n")
    
    demo.launch(share=True)