import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        prompt = "\n".join(self.system_prompt)  # Join system prompts into a single string
        prompt += f'\n\nQuery: "{query}"\nAnswer:<end_of_turn>\n<start_of_turn>model\n'
        return prompt
        
def load_model_and_tokenizer(model_path):
    """
    Load the fine-tuned model and tokenizer from the specified path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    """
    Generate a response from the fine-tuned model given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned causal language model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling.")
    
    args = parser.parse_args()
    
    print("Loading the fine-tuned model...")
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    print("Model loaded successfully!")
    
    print("Generating response...")
    response = generate_response(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
    
    print("\nGenerated Response:")
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main()