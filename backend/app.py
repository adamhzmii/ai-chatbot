import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = Flask(__name__)
CORS(app)

# load and chunk the knowledge base
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"knowledge base loaded — {len(chunks)} chunks")

# using local HuggingFace embeddings — no API cost
print("building vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
print("vector store ready")


def retrieve_context(message: str) -> str:
    # find the 3 chunks most semantically similar to the user's message
    # this is the RAG retrieval step
    docs = vector_store.similarity_search(message, k=3)
    return "\n\n".join([doc.page_content for doc in docs])


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": len(chunks)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)