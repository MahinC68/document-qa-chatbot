import { useState } from "react";

// ── TypingIndicator ───────────────────────────────────────────────────────────
// Three pulsing dots rendered in an assistant bubble while a response is pending
export function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="bubble assistant">
        <span className="typing-dots">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </span>
      </div>
    </div>
  );
}

// ── MessageBubble ─────────────────────────────────────────────────────────────
// Renders a single message in the chat history.
//   - User messages: right-aligned, blue background
//   - Assistant messages: left-aligned, grey background, optional sources
export default function MessageBubble({ message }) {
  // Controls whether the collapsible sources list is open
  const [sourcesOpen, setSourcesOpen] = useState(false);

  // Only assistant messages can have sources
  const hasSources =
    message.role === "assistant" && message.sources?.length > 0;

  return (
    <div className={`message-row ${message.role}`}>
      <div className={`bubble ${message.role}`}>

        {/* Message text content */}
        {message.content}

        {/* Collapsible sources section — assistant only, non-empty sources array */}
        {hasSources && (
          <div className="sources-section">
            {/* Toggle button shows count and open/close indicator */}
            <button
              className="sources-toggle"
              onClick={() => setSourcesOpen((prev) => !prev)}
            >
              Sources ({message.sources.length})
              <span className="sources-chevron">{sourcesOpen ? "▲" : "▼"}</span>
            </button>

            {/* Source chips — each shows document name and page number */}
            {sourcesOpen && (
              <div className="sources-list">
                {message.sources.map((s, i) => (
                  <span key={i} className="source-chip">
                    {s.document} · p{s.page + 1}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
