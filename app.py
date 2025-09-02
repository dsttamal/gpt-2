from flask import Flask, render_template, request, jsonify, stream_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json

app = Flask(__name__)

# Use local 355M model directory (transformers format) with absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models", "gpt2-355M-transformers")

# Global variables to store the model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the local GPT-2 model once when the app starts"""
    global model, tokenizer, device
    
    print("Loading GPT-2 355M model...")
    
    try:
        # Try to load the local model
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        print(f"Successfully loaded local model from {model_dir}")
    except Exception as e:
        print(f"Could not load local model: {e}")
        print("Using GPT2-medium from Hugging Face instead...")
        
        # Fallback to downloading GPT2-medium
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        
        # Save the model locally for future use
        local_save_dir = os.path.join(base_dir, "models", "gpt2-355M-transformers")
        os.makedirs(local_save_dir, exist_ok=True)
        tokenizer.save_pretrained(local_save_dir)
        model.save_pretrained(local_save_dir)
        print(f"355M model saved to {local_save_dir} for future local use")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on device: {device}")

def generate_text(prompt, max_length=120, temperature=0.9, top_p=0.9, top_k=50):
    """Generate text using the loaded model"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        return "Model not loaded"
    
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
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,  # Avoid repetition
            early_stopping=True
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

@app.route('/')
def index():
    """Main page with the text generation interface"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint for text generation"""
    try:
        data = request.get_json()
        
        prompt = data.get('prompt', '')
        max_length = int(data.get('max_length', 120))
        temperature = float(data.get('temperature', 0.9))
        top_p = float(data.get('top_p', 0.9))
        top_k = int(data.get('top_k', 50))
        
        if not prompt.strip():
            return jsonify({'error': 'Please provide a prompt'}), 400
        
        # Generate text
        generated_text = generate_text(prompt, max_length, temperature, top_p, top_k)
        
        return jsonify({
            'generated_text': generated_text,
            'prompt': prompt,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check if the model is loaded"""
    global model, tokenizer, device
    return jsonify({
        'model_loaded': model is not None,
        'device': device,
        'model_dir': model_dir
    })

if __name__ == '__main__':
    # Load the model when the app starts
    load_model()
    
    # Run the Flask app
    print("Starting web interface...")
    print("Open your browser to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
