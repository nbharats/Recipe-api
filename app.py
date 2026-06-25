from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Initialize Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN
)

@app.route("/api/recipe", methods=["POST"])
def generate_recipe():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        ingredients = data.get("ingredients", [])

        if not ingredients:
            return jsonify({
                "error": "No ingredients provided"
            }), 400

        prompt = f"""
        Create a detailed recipe using these ingredients:

        {", ".join(ingredients)}

        Please return:
        1. Recipe Title
        2. Preparation Time
        3. Ingredients List
        4. Step-by-Step Instructions
        5. Serving Suggestions

        Format the response in Markdown.
        """

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional chef who creates clear, delicious recipes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024
        )

        recipe = response.choices[0].message.content

        return jsonify({
            "recipe": recipe
        })

    except Exception as e:
        print("FULL ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running"
    })


if __name__ == "__main__":
    app.run()