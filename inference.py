#!/usr/bin/env python3
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# -----------------------
# Prompts
# -----------------------
llm_cot_prompt_gemma2 = (
    "<bos><start_of_turn>user\n"
    "For the given query including a meal description, think step by step as follows:\n"
    "1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.\n"
    "2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size.\n"
    '3. Respond with a dictionary object containing the total carbohydrates in grams as follows:\n'
    '{{"total_carbohydrates": total grams of carbohydrates for the serving}}\n'
    'For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don\'t know the answer, set the value of "total_carbohydrates" to -1.\n\n'
    "Follow the format of the following examples when answering\n\n"
    'Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."\n'
    "Answer: Let's think step by step.\n"
    "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
    "1 cup of oatmeal has 27g carbs.\n"
    "1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n"
    "1 glass of orange juice has 26g carbs.\n"
    "So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5\n"
    'Output: {{"total_carbohydrates": 66.5}}\n\n'
    'Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."\n'
    "Answer: Let's think step by step.\n"
    "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
    "Scrambled eggs made with 2 eggs has 2g carbs.\n"
    "1 toast has 13g carbs.\n"
    "So the total grams of carbs in the meal = (2 + 13) = 15\n"
    'Output: {{"total_carbohydrates": 15}}\n\n'
    'Query: "Half a peanut butter and jelly sandwich."\n'
    "Answer: Let's think step by step.\n"
    "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
    "1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (25.3*(1/2)) = 25.3g carbs\n"
    "So the total grams of carbs in the meal = 25.3\n"
    'Output: {{"total_carbohydrates": 25.3}}\n\n'
    'Query: {query}\n'
    "Answer: Let's think step by step.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# -----------------------
# Helper to clean the model output
# -----------------------
def extract_model_response(decoded_text: str) -> str:
    """
    Extracts only the content between <start_of_turn>model and <end_of_turn>.
    Removes special tokens like <bos>, <eos>, etc.
    """
    # 1. Find the portion inside <start_of_turn>model ... <end_of_turn>
    match = re.search(r"<start_of_turn>model(.*?)<end_of_turn>", decoded_text, re.DOTALL)
    if match:
        snippet = match.group(1)
    else:
        # If the regex didn't find anything, fallback to the entire decoded text
        snippet = decoded_text

    # 2. Remove any other special tokens in angle brackets, e.g. <bos>, <eos>, etc.
    snippet = re.sub(r"<.*?>", "", snippet)
    return snippet.strip()

# -----------------------
# Class-based Inference
# -----------------------
class GemmaInference:
    def __init__(self, model_type, adapter_path=None, local_model_dir=None):
        """
        model_type: '2b' -> 'google/gemma-2-2b-it'
                    '27b' -> 'google/gemma-2-27b-it'
        adapter_path: path to a LoRA adapter (applies only if model_type='27b')
        local_model_dir: local path for the base model
        """
        self.model_type = model_type.lower()
        self.adapter_path = adapter_path
        self.local_model_dir = local_model_dir
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Loads the base model from a local directory if provided, otherwise from HF hub.
        Applies the adapter if needed (27B only).
        """
        if self.model_type == "2b":
            default_repo = "google/gemma-2-2b-it"
        elif self.model_type == "27b":
            default_repo = "google/gemma-2-27b-it"
        else:
            raise ValueError("model_type must be either '2b' or '27b'.")

        # If user provided a local directory path, use that; otherwise use HF hub name
        if self.local_model_dir and os.path.isdir(self.local_model_dir):
            base_model_path = self.local_model_dir
        else:
            base_model_path = default_repo

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",           # Put layers on GPU if available
            torch_dtype=torch.bfloat16   # Adjust if needed
        )

        # Apply LoRA adapter if using 27B and an adapter path is provided
        if self.model_type == "27b" and self.adapter_path and os.path.isdir(self.adapter_path):
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

    def get_prompt(self, query: str) -> str:
        return llm_cot_prompt_gemma2.format(query=query)

    def generate(self, prompt: str, max_new_tokens=4096, do_sample=False, temperature=0.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        # Decode
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return decoded

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Inference for Gemma 2B or 27B with optional LoRA adapter.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Which model to load: '2b' or '27b'.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint (for 27B model only).")
    parser.add_argument("--local_model_dir", type=str, default=None,
                        help="Local path where the base model is downloaded. If not set, will pull from HF Hub.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="User prompt (meal description).")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0 for deterministic).")
    parser.add_argument("--do_sample", action="store_true",
                        help="Flag to allow sampling. By default, sampling is off.")

    args = parser.parse_args()

    gemma = GemmaInference(
        model_type=args.model_type,
        adapter_path=args.adapter_path,
        local_model_dir=args.local_model_dir
    )
    gemma.load_model()

    final_prompt = gemma.get_prompt(args.prompt)
    raw_output = gemma.generate(
        prompt=final_prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature
    )

    # Clean and extract only the final chunk we care about
    cleaned_output = extract_model_response(raw_output)

    # Print the cleaned text (without <end_of_turn>, <bos>, etc.)
    print(cleaned_output)


if __name__ == "__main__":
    main()