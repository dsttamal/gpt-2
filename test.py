from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Use local 355M model directory (transformers format)
model_dir = "models/gpt2-355M-transformers"

def load_local_model():
    """
    Load the local GPT-2 model files.
    Note: This requires the model to be in Transformers format.
    If you have TensorFlow checkpoints, you'll need to convert them first.
    """
    try:
        # Try to load as if it's already in transformers format
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        print(f"Successfully loaded local model from {model_dir}")
        return tokenizer, model
    except Exception as e:
        print(f"Could not load local model: {e}")
        print("The local 355M model appears to be in TensorFlow format.")
        print("Using GPT2-medium from Hugging Face instead...")
        
        # Fallback to downloading GPT2-medium (355M parameters) but save it locally for future use
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        
        # Save the model locally for future use
        local_save_dir = "models/gpt2-355M-transformers"
        os.makedirs(local_save_dir, exist_ok=True)
        tokenizer.save_pretrained(local_save_dir)
        model.save_pretrained(local_save_dir)
        print(f"355M model saved to {local_save_dir} for future local use")
        
        return tokenizer, model

def generate_text(prompt, max_length=120, temperature=0.9, top_p=0.9):
    """Generate text using the local model"""
    tokenizer, model = load_local_model()
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Run the generation
if __name__ == "__main__":
    prompt = "What is the meaning of life?"
    print(f"Prompt: {prompt}")
    print(f"Generated text:")
    print("-" * 50)
    
    generated = generate_text(prompt)
    print(generated)
