import os
from vectorstore import generateEmbeddings
from semantic_search import retrieveRelevantDocs
from generation import explainSelectedVerses
from flask import Flask, request, jsonify 
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load and prepare once
# gitaDf = loadAndProcessGita()
INDEX_NAME = "gita-index"
df = pd.read_csv(os.path.join("data", "structured_gita.csv"))
vectorStore = generateEmbeddings(df, INDEX_NAME)

@app.route("/", methods=["GET"])
def home():
    return "✨ Gita API is running. Use /api/gita with POST requests."

@app.route("/api/gita", methods=["POST"])
def gita():
    data = request.json
    question = data.get("question")
    print(f"Question received: {question}")
    topDocs = retrieveRelevantDocs(question, vectorStore)
    verses = explainSelectedVerses(topDocs)

    return jsonify({
        "verses": [
            {
                "verse_no": v["verse_no"],
                "sanskrit_text": v["sanskrit_text"],
                "translation": v["translation"],
                "explanation": v["explanation"]
            } for v in verses
        ],
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)

