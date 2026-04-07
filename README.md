# AI Customer Support Chatbot

An intelligent customer support assistant with a full RAG pipeline, intent classification, and streaming responses — built to simulate real-world Agent Assist and Voicebot systems.

## Features
- **Intent Classification** — routes each message into a category (Billing, Technical, Account, Storage, Teams) before generating a response
- **RAG Pipeline** — embeds queries locally using HuggingFace, searches a FAISS vector store, and injects the top 3 relevant knowledge base chunks into the prompt
- **Streaming Responses** — streams tokens to the frontend via Server-Sent Events
- **Conversation History** — maintains context across multiple turns
- **Full Stack** — React frontend, Flask REST API, Docker Compose deployment

## Tech Stack
| Layer | Technology |
|---|---|
| Frontend | React |
| Backend | Python, Flask |
| LLM | Groq API (Llama 3.1) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector Store | FAISS |
| Orchestration | LangChain |
| Deployment | Docker Compose |

## How It Works
1. User sends a message from the React frontend
2. Flask backend classifies intent using Groq's Llama 3.1
3. The message is embedded locally and used to search FAISS for the 3 most relevant knowledge base chunks
4. Retrieved context + conversation history are injected into the system prompt
5. Groq streams the response back token by token
6. Frontend displays the intent badge and renders the streamed text in real time

## Running Locally
**Prerequisites:** Python 3.12+, Node.js 22+, Groq API key (free at console.groq.com)
```bash
git clone https://github.com/adamhzmii/ai-chatbot.git
cd ai-chatbot

# backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python app.py

# frontend (new terminal tab)
cd frontend
npm install
npm start
```

App runs at `http://localhost:3000`

## Architecture
```
User → React UI → Flask API → Intent Classifier (Groq)
                            → FAISS Vector Search → Knowledge Base
                            → LLM Response (Groq, streamed)
                            → React UI (token by token)
```