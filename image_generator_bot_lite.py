import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")

# Force CPU usage for compatibility
device = "cpu"
dtype = torch.float32

print("🎨 AI Image Generator - CPU Version")
print("Note: CPU generation is slower but works on any machine")

# Global variable to store pipeline
pipe = None

def load_model():
    """Load Stable Diffusion model optimized for CPU"""
    global pipe
    
    if pipe is not None:
        return pipe
    
    print("Loading model... This will take a moment on first run.")
    print("Model will be cached for future use (~5GB download).")
    
    # Load model with CPU optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True
    )
    
    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable CPU optimizations
    pipe.enable_attention_slicing()
    
    print("✅ Model loaded successfully!")
    return pipe

def generate_image(prompt, style, quality):
    """Generate an image with style presets"""
    
    if not prompt:
        return None, "Please enter a description"
    
    try:
        # Load model
        pipeline = load_model()
        
        # Style presets
        style_prompts = {
            "Realistic": "highly detailed, photorealistic, 8k",
            "Artistic": "artistic, painting, colorful, creative",
            "Anime": "anime style, manga, japanese art",
            "Fantasy": "fantasy art, magical, ethereal",
            "Minimalist": "minimalist, simple, clean design"
        }
        
        # Quality settings
        quality_settings = {
            "Fast (Lower Quality)": {"steps": 15, "size": 256},
            "Balanced": {"steps": 25, "size": 384},
            "High Quality (Slow)": {"steps": 35, "size": 512}
        }
        
        settings = quality_settings[quality]
        
        # Combine prompt with style
        full_prompt = f"{prompt}, {style_prompts[style]}"
        
        # Generate image
        start_time = time.time()
        
        print(f"Generating {settings['size']}x{settings['size']} image with {settings['steps']} steps...")
        
        image = pipeline(
            prompt=full_prompt,
            negative_prompt="low quality, blurry, distorted, ugly",
            num_inference_steps=settings['steps'],
            guidance_scale=7.5,
            height=settings['size'],
            width=settings['size']
        ).images[0]
        
        # Upscale smaller images for better display
        if settings['size'] < 512:
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        generation_time = time.time() - start_time
        
        info = f"✅ Generated in {generation_time:.1f} seconds | Style: {style} | Quality: {quality}"
        
        return image, info
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="🎨 Free AI Image Generator") as demo:
    gr.Markdown("""
    # 🎨 Free AI Image Generator
    
    Generate images using Stable Diffusion - completely free and runs on your computer!
    
    **How to use:**
    1. Describe what you want to see
    2. Choose an art style
    3. Select quality (higher quality = slower)
    4. Click Generate!
    """)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="What do you want to create?",
                placeholder="Example: A cozy cabin in the mountains during sunset",
                lines=3
            )
            
            with gr.Row():
                style = gr.Radio(
                    choices=["Realistic", "Artistic", "Anime", "Fantasy", "Minimalist"],
                    value="Artistic",
                    label="Style"
                )
            
            quality = gr.Radio(
                choices=["Fast (Lower Quality)", "Balanced", "High Quality (Slow)"],
                value="Balanced",
                label="Quality/Speed"
            )
            
            generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
            
            info = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column():
            output = gr.Image(label="Your Creation")
    
    # Examples
    gr.Examples(
        examples=[
            ["A magical forest with glowing mushrooms", "Fantasy", "Balanced"],
            ["A robot having tea in a garden", "Artistic", "Fast (Lower Quality)"],
            ["Portrait of a wise old sailor", "Realistic", "High Quality (Slow)"],
            ["Cherry blossoms and mount fuji", "Anime", "Balanced"],
            ["Abstract representation of music", "Minimalist", "Fast (Lower Quality)"],
        ],
        inputs=[prompt, style, quality],
        outputs=[output, info],
        fn=generate_image,
        cache_examples=False
    )
    
    # Event handlers
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, style, quality],
        outputs=[output, info]
    )
    
    prompt.submit(
        fn=generate_image,
        inputs=[prompt, style, quality],
        outputs=[output, info]
    )
    
    gr.Markdown("""
    ---
    **Note:** First run will download the model (~5GB). Future runs will be faster.
    
    **Tips for better results:**
    - Be specific and descriptive
    - Mention colors, lighting, and mood
    - Try different styles to see what works best
    
    **System Requirements:**
    - 8GB+ RAM recommended
    - ~6GB disk space for model
    - Works on CPU (no GPU required)
    """)

if __name__ == "__main__":
    print("\n🚀 Starting Image Generator...")
    print("📱 Opening in your browser...\n")
    
    # Pre-load model
    print("Pre-loading model for faster first generation...")
    load_model()
    
    # Launch the app
    demo.launch(share=True)