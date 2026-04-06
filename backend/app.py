import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

# initialise Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# load and chunk the knowledge base
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"knowledge base loaded — {len(chunks)} chunks")

# build vector store using local HuggingFace embeddings (free, runs on your Mac)
print("building vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
print("vector store ready")


def classify_intent(message: str) -> str:
    # sends the user message to Groq and asks it to classify into one category
    # this is exactly what a real voicebot routing system does before handling a call
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user message into exactly one of these categories: "
                    "BILLING, TECHNICAL, ACCOUNT, STORAGE, TEAMS, GENERAL. "
                    "Reply with only the category word, nothing else."
                )
            },
            {"role": "user", "content": message}
        ],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()


def retrieve_context(message: str) -> str:
    # find the 3 most semantically similar chunks to the user's message
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