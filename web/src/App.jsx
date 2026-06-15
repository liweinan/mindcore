import { useState } from "react";
import { sendChat } from "./api.js";

export default function App() {
  const [userId, setUserId] = useState("demo");
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    const trimmedUserId = userId.trim();
    const trimmedMessage = message.trim();
    if (!trimmedUserId || !trimmedMessage) {
      setError("请填写用户 ID 和消息");
      return;
    }

    setLoading(true);
    try {
      const data = await sendChat({
        userId: trimmedUserId,
        message: trimmedMessage,
        sessionId,
      });
      setSessionId(data.session_id);
      setResult(data);
    } catch (err) {
      setError(err.message || "网络错误");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header>
        <h1>MindCore 学习对话</h1>
        <p className="subtitle">
          React 前端示例，调用 <code>POST /v1/chat</code>
        </p>
      </header>

      <form className="card" onSubmit={handleSubmit} data-testid="chat-form">
        <label htmlFor="user-id">用户 ID</label>
        <input
          id="user-id"
          data-testid="user-id"
          name="user_id"
          type="text"
          value={userId}
          maxLength={64}
          required
          onChange={(event) => setUserId(event.target.value)}
        />

        <label htmlFor="message">消息</label>
        <textarea
          id="message"
          data-testid="message"
          name="message"
          rows={4}
          maxLength={2000}
          placeholder="输入你想说的话…"
          required
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />

        <button id="send-btn" data-testid="send-btn" type="submit" disabled={loading}>
          {loading ? "发送中…" : "发送"}
        </button>
      </form>

      {result ? (
        <section className="card" data-testid="result" aria-live="polite">
          <h2>回复</h2>
          <p className="reply" data-testid="reply">
            {result.reply}
          </p>
          <dl className="meta">
            <div>
              <dt>风险等级</dt>
              <dd data-testid="risk-level">{result.risk_level ?? "-"}</dd>
            </div>
            <div>
              <dt>置信度</dt>
              <dd data-testid="confidence">{result.confidence ?? "-"}</dd>
            </div>
            <div>
              <dt>会话 ID</dt>
              <dd data-testid="session-id">{result.session_id || "-"}</dd>
            </div>
            <div>
              <dt>推理耗时</dt>
              <dd data-testid="inference-ms">
                {result.inference_time_ms != null ? `${result.inference_time_ms} ms` : "-"}
              </dd>
            </div>
          </dl>
        </section>
      ) : null}

      {error ? (
        <p className="error" data-testid="error" role="alert">
          {error}
        </p>
      ) : null}
    </main>
  );
}
