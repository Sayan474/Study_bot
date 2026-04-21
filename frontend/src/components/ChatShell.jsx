import { useMemo, useState } from "react";
import { askQuestion } from "../api";

export default function ChatShell({ user, messages, setMessages, onLogout }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const greeting = useMemo(() => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 17) return "Good afternoon";
    return "Good evening";
  }, []);

  function normalizeMessage(text) {
    if (!text) {
      return "";
    }
    return text.replace(/&nbsp;/g, " ");
  }

  async function submitQuestion(event) {
    event.preventDefault();
    if (!question.trim() || loading) {
      return;
    }

    const nextQuestion = question.trim();
    setQuestion("");
    setError("");
    setLoading(true);

    const userMessage = { role: "user", message: nextQuestion };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await askQuestion({
        user_id: user.user_id, 
        question: nextQuestion,
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", message: response.response },
      ]);
    } catch (requestError) {
      setMessages((prev) => prev.slice(0, -1));
      const message = requestError.message || "Unable to get response right now";
      if (message.toLowerCase().includes("user not found")) {
        onLogout();
        return;
      }
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-layout">
      <aside className="chat-side">
        <h2>{greeting}, {user.username}</h2>
        <p>Ask anything academic and keep your progress tied to your profile.</p>
        <div className="id-tag">User ID: {user.user_id}</div>
        <button type="button" className="secondary-btn" onClick={onLogout}>
          Logout
        </button>
      </aside>

      <section className="chat-main">
        <div className="chat-history">
          {messages.length === 0 && (
            <div className="empty-chat">
              <h3>Start your first study session</h3>
              <p>Try: Explain Newton's laws with real life examples.</p>
            </div>
          )}

          {messages.map((item, index) => (
            <article
              key={`${item.role}-${index}`}
              className={item.role === "user" ? "bubble user" : "bubble assistant"}
            >
              <span className="role">{item.role === "user" ? "You" : "Study Bot"}</span>
              <p className="message-text">{normalizeMessage(item.message)}</p>
            </article>
          ))}
        </div>

        <form onSubmit={submitQuestion} className="chat-form">
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about science, math, programming, or exams..."
            disabled={loading}
          />
          <button type="submit" className="action-btn" disabled={loading}>
            {loading ? "Thinking..." : "Send"}
          </button>
        </form>

        {error && <p className="form-error">{error}</p>}
      </section>
    </div>
  );
}
