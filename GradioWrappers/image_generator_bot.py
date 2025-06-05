import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import time

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}")

# Available models
MODELS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "OpenJourney": "prompthero/openjourney",
    "Dreamlike Photoreal": "dreamlike-art/dreamlike-photoreal-2.0",
}

# Global variable to store current pipeline
current_pipe = None
current_model = None

def load_model(model_name):
    """Load the selected model"""
    global current_pipe, current_model
    
    if current_model == model_name:
        return current_pipe
    
    print(f"Loading {model_name}...")
    
    # Clear previous model from memory
    if current_pipe is not None:
        del current_pipe
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Load new model
    model_id = MODELS[model_name]
    current_pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Optimize for speed
    current_pipe.scheduler = DPMSolverMultistepScheduler.from_config(current_pipe.scheduler.config)
    current_pipe = current_pipe.to(device)
    
    # Enable memory efficient attention if available
    if device == "cuda":
        try:
            current_pipe.enable_attention_slicing()
        except:
            pass
    
    current_model = model_name
    print(f"Model {model_name} loaded successfully!")
    
    return current_pipe

def generate_image(prompt, negative_prompt, model_name, num_steps, guidance_scale, seed):
    """Generate an image based on the prompt"""
    
    if not prompt:
        return None, "Please enter a prompt"
    
    try:
        # Load model if needed
        pipe = load_model(model_name)
        
        # Set seed for reproducibility
        generator = torch.Generator(device=device)
        if seed != -1:
            generator = generator.manual_seed(seed)
        
        # Generate image
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=512,
            width=512
        ).images[0]
        
        generation_time = time.time() - start_time
        
        info = f"Generated in {generation_time:.2f} seconds using {model_name}"
        
        return image, info
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_ui():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="🎨 AI Image Generator") as demo:
        gr.Markdown("""
        # 🎨 AI Image Generator
        
        Generate images using open-source Stable Diffusion models. Completely free and runs locally!
        
        **Tips:**
        - Be descriptive with your prompts
        - Use negative prompts to exclude unwanted elements
        - Lower steps = faster generation but lower quality
        - Higher guidance scale = more adherence to prompt
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful sunset over mountains, digital art, highly detailed...",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (what to avoid)",
                    placeholder="low quality, blurry, distorted...",
                    lines=2
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Stable Diffusion v1.5",
                        label="Model"
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Number of Steps"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                
                info_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                output_image = gr.Image(label="Generated Image", type="pil")
        
        # Examples
        gr.Examples(
            examples=[
                ["A cozy coffee shop in autumn, warm lighting, watercolor style", "blurry, dark", "Stable Diffusion v1.5", 25, 7.5, -1],
                ["Futuristic city with flying cars, cyberpunk style, neon lights", "old, rustic", "Stable Diffusion v1.5", 30, 8.0, -1],
                ["Portrait of a wise wizard, fantasy art, detailed", "modern, photograph", "OpenJourney", 25, 7.5, -1],
                ["Tropical beach paradise, crystal clear water, sunset", "cold, winter", "Dreamlike Photoreal", 25, 7.5, -1],
            ],
            inputs=[prompt, negative_prompt, model_dropdown, num_steps, guidance_scale, seed],
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, model_dropdown, num_steps, guidance_scale, seed],
            outputs=[output_image, info_text]
        )
        
        # Keyboard shortcut - Enter to generate
        prompt.submit(
            fn=generate_image,
            inputs=[prompt, negative_prompt, model_dropdown, num_steps, guidance_scale, seed],
            outputs=[output_image, info_text]
        )
    
    return demo

if __name__ == "__main__":
    # Check available memory
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Running on CPU - generation will be slower")
    
    # Create and launch the app
    print("\nStarting Gradio interface...")
    demo = create_ui()
    demo.launch(share=True)