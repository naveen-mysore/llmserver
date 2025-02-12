#!/usr/bin/env python3
import argparse
import logging
import os
import re
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

###############################################################################
# Prompts
###############################################################################
llm_cot_prompt_gemma2 = (
    "<bos><start_of_turn>user\n"
    "For the given query including a meal description, think step by step as follows:\n"
    "1. Parse the meal description into discrete food or beverage items along with their serving size. "
    "If the serving size of any item in the meal is not specified, assume it is a single standard serving "
    "based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate "
    "to the item name and serving size.\n"
    "2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for "
    "the specific serving size.\n"
    '3. Respond with a dictionary object containing the total carbohydrates in grams as follows:\n'
    '{{"total_carbohydrates": total grams of carbohydrates for the serving}}\n'
    'For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. '
    'If you don\'t know the answer, set the value of "total_carbohydrates" to -1.\n\n'
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

###############################################################################
# Helper Function: Extract relevant chunk from the raw model output
###############################################################################
def extract_model_response(decoded_text: str) -> str:
    """
    Extracts only the content between <start_of_turn>model and <end_of_turn>,
    and removes special tokens like <bos>, <eos>, etc.
    """
    # 1. Regex to capture text within <start_of_turn>model ... <end_of_turn>
    match = re.search(r"<start_of_turn>model(.*?)<end_of_turn>", decoded_text, re.DOTALL)
    if match:
        snippet = match.group(1)
    else:
        # If not found, fallback to entire decoded text
        snippet = decoded_text

    # 2. Remove angle-bracket tokens (e.g. <bos>, <eos>, etc.)
    snippet = re.sub(r"<.*?>", "", snippet)
    return snippet.strip()

###############################################################################
# GemmaInference Class
###############################################################################
class GemmaInference:
    def __init__(self, model_type, adapter_path=None, local_model_dir=None, max_new_tokens=4096, do_sample=False, temperature=0.0):
        """
        model_type: '2b' -> 'google/gemma-2-2b-it'
                    '27b' -> 'google/gemma-2-27b-it'
        adapter_path: path to a LoRA adapter (only used if model_type='27b')
        local_model_dir: local path to the base model (if you have it downloaded)
        max_new_tokens, do_sample, temperature: generation hyperparameters
        """
        self.model_type = model_type.lower()
        self.adapter_path = adapter_path
        self.local_model_dir = local_model_dir

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Loads the base model from a local directory if provided, otherwise from HF hub,
        and applies the adapter if needed (only for 27B).
        """
        if self.model_type == "2b":
            default_repo = "google/gemma-2-2b-it"
        elif self.model_type == "27b":
            default_repo = "google/gemma-2-27b-it"
        else:
            raise ValueError("model_type must be '2b' or '27b'.")

        # Determine model path (local vs. HF hub)
        if self.local_model_dir and os.path.isdir(self.local_model_dir):
            base_model_path = self.local_model_dir
        else:
            base_model_path = default_repo

        logging.info(f"Loading tokenizer from: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

        logging.info(f"Loading model from: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

        # Apply LoRA adapter if needed
        if self.model_type == "27b" and self.adapter_path and os.path.isdir(self.adapter_path):
            logging.info(f"Applying LoRA adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        logging.info("Model and tokenizer loaded successfully!")

    def get_prompt(self, query: str) -> str:
        """
        Inserts the user query into the system prompt.
        """
        return llm_cot_prompt_gemma2.format(query=query)

    def generate(self, prompt: str) -> str:
        """
        Generates text given a prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return decoded

###############################################################################
# Flask App
###############################################################################
app = Flask(__name__)

# Logging to file + console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)

# We'll define 'gemma_inference' after parsing cmd args in main()
gemma_inference = None

@app.route("/", methods=["POST"])
def handle_request():
    try:
        # Parse incoming JSON
        request_data = request.get_json()
        if request_data is None:
            response = {"error": "Invalid JSON payload"}
            logging.warning(f"Invalid JSON received: {request.data}")
            return jsonify(response), 400

        # Log the received data
        logging.info(f"Received request: {request_data}")

        # Build prompt + generate raw model output
        prompt = gemma_inference.get_prompt(request_data["prompt"])
        raw_output = gemma_inference.generate(prompt)
        cleaned_output = extract_model_response(raw_output)

        # Log response before sending
        logging.info(f"Response sent: {cleaned_output}")

        return jsonify({"response": cleaned_output})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

###############################################################################
# Main Entry
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Inference Server")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Which model to load: '2b' or '27b'.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (for 27B model only).")
    parser.add_argument("--local_model_dir", type=str, default=None,
                        help="Local path for base model if you have it already.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling. By default sampling is off.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on.")

    args = parser.parse_args()

    # Initialize GemmaInference
    gemma_inference = GemmaInference(
        model_type=args.model_type,
        adapter_path=args.adapter_path,
        local_model_dir=args.local_model_dir,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature
    )
    gemma_inference.load_model()

    # Run the Flask app
    app.run(host="0.0.0.0", port=args.port)