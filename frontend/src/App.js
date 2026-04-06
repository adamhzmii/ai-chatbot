import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Zap } from "lucide-react";
import "./App.css";

const INTENT_COLORS = {
  BILLING: "#f59e0b",
  TECHNICAL: "#3b82f6",
  ACCOUNT: "#8b5cf6",
  STORAGE: "#10b981",
  TEAMS: "#ec4899",
  GENERAL: "#6b7280",
};

function Message({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "bot-row"}`}>
      <div className="avatar">
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="message-content">
        {msg.intent && (
          <span
            className="intent-badge"
            style={{ backgroundColor: INTENT_COLORS[msg.intent] || "#6b7280" }}
          >
            <Zap size={10} />
            {msg.intent}
          </span>
        )}
        <div className={`bubble ${isUser ? "user-bubble" : "bot-bubble"}`}>
          {msg.content || <span className="typing">●●●</span>}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I'm the Nexus support assistant. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    // add user message to chat
    const userMsg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // add an empty assistant message that we'll stream into
    setMessages((prev) => [...prev, { role: "assistant", content: "", intent: null }]);

    try {
      const res = await fetch("http://localhost:5001/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-6),
        }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let detectedIntent = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const chunk = line.slice(6);

          if (chunk === "[DONE]") break;

          // check if this chunk is the intent signal
          if (chunk.startsWith("[INTENT:")) {
            detectedIntent = chunk.slice(8, -1);
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1].intent = detectedIntent;
              return updated;
            });
            continue;
          }

          // stream the text into the last message
          const text = chunk.replace(/\\n/g, "\n");
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1].content += text;
            return updated;
          });
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].content =
          "Sorry, something went wrong. Please try again.";
        return updated;
      });
    }

    setLoading(false);
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <Bot size={20} />
        <span>Nexus Support Assistant</span>
        <span className="header-sub">AI-powered · RAG · Intent routing</span>
      </header>

      <div className="messages">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Type a message... (Enter to send)"
          rows={1}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}