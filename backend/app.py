import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = Flask(__name__)
CORS(app)

# load the knowledge base text file and split into chunks
# we split into chunks because the AI has a limited context window
# smaller chunks also mean more precise retrieval later
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"knowledge base loaded — {len(chunks)} chunks")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": len(chunks)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)