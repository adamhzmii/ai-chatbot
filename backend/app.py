import os
from flask import Flask, jsonify, request, Response, stream_with_context
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
        model="llama-3.1-8b-instant",
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


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "no message provided"}), 400

    # step 1 — classify what the user is asking about
    intent = classify_intent(user_message)

    # step 2 — retrieve the most relevant knowledge base chunks
    context = retrieve_context(user_message)

    # step 3 — build the system prompt with the retrieved context injected
    # this is what makes it RAG — the AI answers from the knowledge base, not from memory
    system_prompt = f"""You are a helpful customer support assistant for Nexus, a cloud productivity platform.

Use the following knowledge base context to answer the customer's question accurately.
If the answer is not in the context, say you will escalate to a human agent — never make up information.

Intent detected: {intent}

Knowledge Base Context:
{context}

Guidelines:
- Be concise and friendly
- If it is a BILLING or ACCOUNT issue you cannot resolve, offer to escalate to a human agent
- Always end by asking if there is anything else you can help with
"""

    # step 4 — build the message list including recent conversation history
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # step 5 — stream the response back token by token
    # streaming means the frontend can display words as they arrive, like ChatGPT
    def generate():
        # send the intent first so the frontend can display the badge
        yield f"data: [INTENT:{intent}]\n\n"

        stream = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=400,
            temperature=0.4,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                # escape newlines so they survive the SSE transport
                safe = delta.replace("\n", "\\n")
                yield f"data: {safe}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)