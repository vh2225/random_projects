import gradio as gr
import ollama

# Available models
MODELS = ["tinyllama", "llama3.2:1b", "gemma2:2b"]

def chat_with_girlfriend(message, history, girlfriend_description, model):
    """
    Chat with AI girlfriend based on user's description
    
    Args:
        message: User's current message
        history: Chat history
        girlfriend_description: User's ideal girlfriend description
        model: Selected LLM model
    
    Returns:
        AI's response
    """
    # Build system prompt from user's description
    system_prompt = f"""You are roleplaying as someone's ideal girlfriend based on this description: {girlfriend_description}

Stay in character and respond naturally as this person would. Be warm, engaging, and maintain the personality described. Remember previous conversations and build on them.

Important: Try to sound as human as possible and avoid mentioning that you're an AI unless the user specifically asks about it. Respond as a real person would."""
    
    # Convert Gradio history format to Ollama format
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Get response from Ollama
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running with 'ollama serve'"

# Create the Gradio interface
with gr.Blocks(title="AI Girlfriend Chat") as demo:
    gr.Markdown("# 💕 AI Girlfriend Roleplay Chat")
    gr.Markdown("Describe your ideal girlfriend and chat with her!")
    
    with gr.Row():
        with gr.Column(scale=1):
            girlfriend_input = gr.Textbox(
                label="Describe your ideal girlfriend:",
                placeholder="e.g., A caring and intelligent woman who loves books, has a great sense of humor, enjoys cooking together, and is supportive of my goals...",
                lines=4
            )
            model_dropdown = gr.Dropdown(
                choices=MODELS,
                value="gemma2:2b",
                label="Select LLM Model"
            )
    
    with gr.Row():
        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
            bubble_full_width=False
        )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your message:",
            placeholder="Type your message here...",
            lines=1
        )
        send_btn = gr.Button("Send", variant="primary")
    
    def respond(message, chat_history, girlfriend_desc, model_name):
        """Handle the chat interaction"""
        if not girlfriend_desc:
            bot_message = "Please describe your ideal girlfriend first before we can start chatting! 💕"
        else:
            bot_message = chat_with_girlfriend(message, chat_history, girlfriend_desc, model_name)
        
        chat_history.append((message, bot_message))
        return "", chat_history
    
    # Set up event handlers
    msg.submit(respond, [msg, chatbot, girlfriend_input, model_dropdown], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot, girlfriend_input, model_dropdown], [msg, chatbot])
    
    # Add examples
    gr.Examples(
        examples=[
            "Hi! How was your day?",
            "What do you like to do for fun?",
            "Tell me about your dreams",
            "What's your favorite food?",
        ],
        inputs=msg
    )

if __name__ == "__main__":
    # Check if Ollama is available
    try:
        ollama.list()
        print("✅ Ollama is running!")
        
        # Check for required models
        for model in MODELS:
            try:
                ollama.show(model)
                print(f"✅ {model} is available")
            except:
                print(f"⚠️  {model} not found. Pulling...")
                ollama.pull(model)
                print(f"✅ {model} downloaded")
                
    except:
        print("❌ Ollama is not running. Please start it with 'ollama serve' in a terminal")
        
    # Launch the app with public link
    demo.launch(share=True)