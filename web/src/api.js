export async function sendChat({ userId, message, sessionId }) {
  const payload = { user_id: userId, message };
  if (sessionId) {
    payload.session_id = sessionId;
  }

  const response = await fetch("/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || response.statusText || "请求失败";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return data;
}
