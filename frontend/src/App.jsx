import { useState } from "react";
import { v4 as uuidv4 } from "uuid";

const API = "http://localhost:8000";

export default function App() {
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

  return <div />;
}
