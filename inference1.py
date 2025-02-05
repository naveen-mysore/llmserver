import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel  # LoRA support if needed
import os

class GemmaPrompt:
    def __init__(self):
        self.system_prompt = [
            "<bos><start_of_turn>user",
            "For the given query including a meal description, calculate the amount of carbohydrates in grams.",
            "If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA).",
            "Respond with a dictionary object containing the total carbohydrates in grams as follows **without any additional text**:",
            '{{"total_carbohydrates": total grams of carbohydrates for the serving}}',
            "For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text.",
            'If you don\'t know the answer, respond with:\n{{"total_carbohydrates": -1}}.',
            "\nExamples:",
            'Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."',
            'Answer: {{"total_carbohydrates": 66.5}}',
            'Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast"',
            'Answer: {{"total_carbohydrates": 15}}',
            'Query: "Half a peanut butter and jelly sandwich."',
            'Answer: {{"total_carbohydrates": 25.3}}',
        ]

    def user_prompt(self, query):
        """Generate the same prompt format used during fine-tuning."""
        prompt = "\n".join(self.system_prompt)  
        prompt += f'\n\n<start_of_turn>user\nQuery: "{query}"\nAnswer:<end_of_turn>\n<start_of_turn>model\n'
        return prompt
        
def load_model_and_tokenizer(model_path, lora_path=None):
    """
    Load the fine-tuned model and tokenizer from the specified path.
    Uses `bfloat16` because that's how it was fine-tuned.
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    print(f"Loading base model from {model_path} using bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # ✅ Matches fine-tuning
        device_map="auto"
    )

    # If a LoRA checkpoint is provided, apply it
    if lora_path and os.path.exists(lora_path):
        print(f"Applying LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)

        # Debug: Check if LoRA is correctly applied
        print("✅ Trainable Parameters After Loading LoRA:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}: {param.shape}")

    return tokenizer, model

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.1, top_p=0.1):
    """
    Generate a response using HuggingFacePipeline (matches fine-tuning).
    """
    # ✅ Use HuggingFacePipeline (matches fine-tuning script)
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    outputs = text_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,  # ✅ Matches fine-tuning
        top_p=top_p,  # ✅ Matches fine-tuning
        do_sample=True,
        truncation=True
    )

    return outputs[0]['generated_text']

def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned causal language model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA adapter checkpoint (if applicable).")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    
    args = parser.parse_args()
    
    print("Loading the fine-tuned model...")
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.lora_path)
    print("Model loaded successfully!")

    prompt_generator = GemmaPrompt()
    formatted_prompt = prompt_generator.user_prompt(args.prompt)
    
    print("Generating response...")
    response = generate_response(model, tokenizer, formatted_prompt, args.max_new_tokens)
    
    print("\nGenerated Response:")
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main()
