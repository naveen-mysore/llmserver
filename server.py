import logging
from inference import load_model_and_tokenizer, generate_response
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = "/mnt/data/llmserver/model"
tokenizer, model = load_model_and_tokenizer(model_path)

# Configure logging to log to both file and console
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs.txt"),  # Log to file
        logging.StreamHandler()  # Log to console (screen output)
    ]
)

@app.route("/", methods=["POST"])
def handle_request():
    try:
        # Parse incoming JSON payload
        request_data = request.get_json()

        if request_data is None:
            response = {"error": "Invalid JSON payload"}
            logging.warning(f"Invalid JSON received: {request.data}")
            return jsonify(response), 400

        # Log the received data
        logging.info(f"Received request: {request_data}")

        response = generate_response(model, tokenizer, request_data["prompt"])

        # Log response before sending
        logging.info(f"Response sent: {response}")

        return jsonify({"response": response})  # Ensure proper JSON format

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)