import { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import "./index.css";

const API = "http://localhost:8000";

export default function App() {
  // ── core state (logic unchanged) ──────────────────────────────────────────
  const [sessionId, setSessionId] = useState(() => uuidv4());
  const [messages, setMessages] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  async function sendMessage(text) {
    const userMsg = { role: "user", content: text, sources: [] };
    setMessages((prev) => [...prev, userMsg]);

    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, session_id: sessionId }),
    });

    const data = await res.json();
    const assistantMsg = {
      role: "assistant",
      content: data.answer,
      sources: data.sources ?? [],
    };
    setMessages((prev) => [...prev, assistantMsg]);
  }

  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API}/upload`, {
      method: "POST",
      body: form,
    });

    const data = await res.json();
    setUploadedFiles((prev) => [...prev, data.filename]);
  }

  async function newChat() {
    await fetch(`${API}/session/${sessionId}`, { method: "DELETE" });
    setMessages([]);
    setSessionId(uuidv4());
  }

  // ── UI-only state ─────────────────────────────────────────────────────────
  const [input, setInput] = useState("");
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // keep the message list scrolled to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSubmit(e) {
    e?.preventDefault();
    const text = input.trim();
    if (!text) return;
    sendMessage(text);
    setInput("");
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    e.target.value = ""; // reset so the same file can be re-uploaded
  }

  const isChat = messages.length > 0;

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {isChat ? (
        // ── Chat view ───────────────────────────────────────────────────────
        <div className="chat">
          <header className="chat-header">
            <span className="brand-small">DocuChat</span>
            <button className="new-chat-btn" onClick={newChat}>
              New Chat
            </button>
          </header>

          {/* scrollable message column, capped at 700px */}
          <div className="messages-scroll">
            <div className="messages-inner">
              {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`}>
                  <div className={`bubble ${msg.role}`}>
                    {msg.content}

                    {/* source citations shown only on assistant messages */}
                    {msg.role === "assistant" && msg.sources.length > 0 && (
                      <div className="sources">
                        {msg.sources.map((s, j) => (
                          <span key={j} className="source-chip">
                            {s.document} · p{s.page + 1}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {/* invisible anchor so useEffect can scroll here */}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* fixed input bar at the bottom */}
          <div className="chat-input-area">
            <div className="chat-input-inner">
              <button
                className="upload-btn"
                onClick={() => fileInputRef.current?.click()}
                title="Upload PDF"
              >
                <PaperclipIcon />
              </button>
              <form className="input-pill" onSubmit={handleSubmit}>
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about your documents…"
                  autoFocus
                />
                <button type="submit" className="send-btn" aria-label="Send">
                  <SendIcon />
                </button>
              </form>
            </div>
          </div>
        </div>
      ) : (
        // ── Landing view ────────────────────────────────────────────────────
        <div className="landing">
          {/* DocuChat title + search bar + upload, centred in the white area */}
          <div className="landing-center">
            <h1 className="brand-title">DocuChat</h1>
            <form className="input-pill" onSubmit={handleSubmit}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything about your documents…"
                autoFocus
              />
              <button type="submit" className="send-btn" aria-label="Send">
                <SendIcon />
              </button>
            </form>

            <button
              className="upload-btn-landing"
              onClick={() => fileInputRef.current?.click()}
            >
              <PaperclipIcon />
              Upload PDF
            </button>

            {/* show filenames once a PDF has been uploaded */}
            {uploadedFiles.length > 0 && (
              <div className="uploaded-list">
                {uploadedFiles.map((f, i) => (
                  <span key={i} className="file-chip">
                    {f}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* gradient semicircle dome — tagline sits inside the blue glow */}
          <div className="semicircle">
            <p className="brand-tagline">Ask anything about your documents</p>
          </div>
        </div>
      )}

      {/* hidden file input — shared by both views */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
    </div>
  );
}

// ── Icons ─────────────────────────────────────────────────────────────────────
// Inline SVGs avoid an icon-library dependency

function SendIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function PaperclipIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
    </svg>
  );
}
