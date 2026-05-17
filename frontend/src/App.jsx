import { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import "./index.css";
import MessageBubble, { TypingIndicator } from "./MessageBubble";

const API = "https://docuchat-lehq.onrender.com";
const TAGLINE = "Chat with your documents instantly";

export default function App() {
  // ── core state (logic unchanged) ──────────────────────────────────────────
  const [sessionId, setSessionId] = useState(() => uuidv4());
  const [messages, setMessages] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  async function sendMessage(text) {
    const userMsg = { role: "user", content: text, sources: [] };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, session_id: sessionId }),
      });

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources ?? [] },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Could not reach the backend. Make sure the server is running on port 8000.", sources: [] },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    setIsUploading(true);

    try {
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: form,
      });

      const data = await res.json();
      setUploadedFiles((prev) => [...prev, data.filename]);
    } catch {
      alert("Upload failed — make sure the backend server is running on port 8000.");
    } finally {
      setIsUploading(false);
    }
  }

  async function newChat() {
    try {
      await fetch(`${API}/session/${sessionId}`, { method: "DELETE" });
    } catch {
      // session cleanup is best-effort; proceed regardless
    }
    setMessages([]);
    setSessionId(uuidv4());
  }

  // ── UI-only state ─────────────────────────────────────────────────────────
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [fileError, setFileError] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  // typing animation for the landing tagline
  const [typedTagline, setTypedTagline] = useState("");
  const [typingDone, setTypingDone] = useState(false);

  useEffect(() => {
    let i = 0;
    const timer = setInterval(() => {
      i += 1;
      setTypedTagline(TAGLINE.slice(0, i));
      if (i === TAGLINE.length) {
        clearInterval(timer);
        setTypingDone(true);
      }
    }, 50);
    return () => clearInterval(timer);
  }, []);
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

    // Block submission on the landing page if no PDF has been uploaded yet
    if (!isChat && uploadedFiles.length === 0) {
      setFileError("Please upload a PDF before asking a question.");
      setTimeout(() => setFileError(""), 3500);
      setInput("");
      return;
    }

    sendMessage(text);
    setInput("");
  }

  function removeFile(index) {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Enforce single-file limit
    if (uploadedFiles.length > 0) {
      setFileError("Only one file can be uploaded at a time. Remove the current file first.");
      setTimeout(() => setFileError(""), 4000);
      e.target.value = "";
      return;
    }

    // Reject anything that isn't a PDF
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setFileError("Only PDF files are supported. Please select a .pdf file.");
      setTimeout(() => setFileError(""), 4000);
      e.target.value = "";
      return;
    }

    setFileError("");
    uploadFile(file);
    e.target.value = "";
  }

  const isChat = messages.length > 0;

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {isChat ? (
        // ── Chat view ───────────────────────────────────────────────────────
        <div className="chat">
          {/* Fixed header: clicking the logo resets to the landing page */}
          <header className="chat-header">
            <button className="chat-brand" onClick={newChat} title="Go to home">
              <LogoMark size={28} />
              <span className="brand-small">DocuChat</span>
            </button>

            {/* Show the active PDF on the right side of the header */}
            {uploadedFiles.length > 0 && (
              <div className="chat-file-badge">
                <DocFileIcon />
                <span className="chat-file-name">{uploadedFiles[0]}</span>
              </div>
            )}
          </header>

          {/* Scrollable message history — capped at 700px, padded so bubbles
              don't disappear behind the fixed input bar at the bottom */}
          <div className="messages-scroll">
            <div className="messages-inner">
              {messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} />
              ))}
              {/* Typing indicator shown while the assistant is responding */}
              {isLoading && <TypingIndicator />}
              {/* Invisible anchor — scrolled into view on every new message */}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Fixed input bar — no upload in chat, that happens on the landing page */}
          <div className="chat-input-area">
            <div className="chat-input-inner">
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
          {/* DocuChat title + tagline + pills + search bar + upload */}
          <div className="landing-center">
            <LogoMark />
            <h1 className="brand-title">DocuChat</h1>
            <p className="landing-tagline">
              {typedTagline}
              {!typingDone && <span className="typing-cursor">|</span>}
            </p>

            <div className="feature-pills">
              <span className="pill">Ask in seconds</span>
              <span className="pill">Source citations</span>
              <span className="pill">No setup needed</span>
            </div>

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

            {/* Upload button → spinner → file chip, mutually exclusive */}
            {isUploading ? (
              <div className="upload-loading">
                <div className="upload-spinner" />
                <span>Processing PDF…</span>
              </div>
            ) : uploadedFiles.length === 0 ? (
              <button
                className="upload-btn-landing"
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadIcon />
                Upload PDF
              </button>
            ) : null}

            {/* Wrong file type warning — auto-clears after 4s */}
            {fileError && <p className="file-type-error">{fileError}</p>}

            {/* Uploaded file chips with remove button */}
            {uploadedFiles.length > 0 && (
              <div className="uploaded-list">
                {uploadedFiles.map((f, i) => (
                  <span key={i} className="file-chip">
                    <span className="file-chip-name">{f}</span>
                    <button
                      className="file-chip-remove"
                      onClick={() => removeFile(i)}
                      title="Remove file"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* gradient semicircle dome */}
          <div className="semicircle">
            <div className="semicircle-content">
              <p className="rim-label">Powered by RAG + LangGraph</p>
              <p className="rim-statement">Answers grounded in your documents</p>
            </div>
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

function LogoMark({ size = 40 }) {
  return (
    <svg
      className="logo-mark"
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      stroke="#2563eb"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="3" y="2" width="21" height="29" rx="2.5" />
      {/* text lines inside the document */}
      <line x1="8" y1="9"  x2="20" y2="9"  strokeWidth="1.4" />
      <line x1="8" y1="14" x2="20" y2="14" strokeWidth="1.4" />
      <line x1="8" y1="19" x2="15" y2="19" strokeWidth="1.4" />
      <rect x="17" y="22" width="19" height="13" rx="4.5" />
      <path d="M21 35 L19 39 L26 35" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

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

function DocFileIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="8" y1="13" x2="16" y2="13" />
      <line x1="8" y1="17" x2="13" y2="17" />
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
