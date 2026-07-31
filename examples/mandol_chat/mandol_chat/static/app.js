const state = {
  currentSessionId: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setStatus(text) {
  $("status").textContent = text;
}

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = content;
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

function renderMemories(containerId, memories) {
  const container = $(containerId);
  container.innerHTML = "";
  if (!memories || memories.length === 0) {
    container.className = "list empty";
    container.textContent = "No memories.";
    return;
  }
  container.className = "list";
  for (const memory of memories) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <strong>${memory.uid}</strong>
      <div><span class="badge">${memory.session_id || "no-session"}</span> score ${Number(memory.score || 0).toFixed(3)}</div>
      <p>${escapeHtml((memory.content || "").slice(0, 240))}</p>
    `;
    container.appendChild(card);
  }
}

function renderSessions(data) {
  state.currentSessionId = data.active_session_id || state.currentSessionId;
  $("currentSession").textContent = state.currentSessionId || "none";
  const container = $("sessionList");
  container.innerHTML = "";
  const sessions = data.all_sessions || [];
  if (sessions.length === 0) {
    container.className = "list empty";
    container.textContent = "No sessions yet.";
    return;
  }
  container.className = "list";
  for (const session of sessions) {
    const card = document.createElement("div");
    card.className = "card";
    const badgeClass = session.is_finalized ? "badge finalized" : "badge";
    card.innerHTML = `
      <strong>${session.session_id}</strong>
      <span class="${badgeClass}">${session.is_finalized ? "finalized" : "active"}</span>
      <p>units: ${session.unit_count || 0}</p>
      <p>reason: ${session.auto_session_reason || "n/a"} confidence: ${session.auto_session_confidence || "n/a"}</p>
    `;
    container.appendChild(card);
  }
}

async function refreshHealth() {
  const health = await api("/api/health");
  setStatus(`Backend ok · LLM ${health.llm_mode} · embedding ${health.real_embedding ? "real" : "mock"}`);
  state.currentSessionId = health.active_session_id || state.currentSessionId;
  $("currentSession").textContent = state.currentSessionId || "none";
}

async function refreshSessions() {
  const data = await api("/api/sessions");
  renderSessions(data);
}

async function sendMessage(event) {
  event.preventDefault();
  const input = $("messageInput");
  const message = input.value.trim();
  if (!message) return;
  $("sendButton").disabled = true;
  addMessage("user", message);
  input.value = "";
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, user_id: "demo_user", top_k: 5 }),
    });
    addMessage("assistant", data.assistant_message);
    state.currentSessionId = data.session_id;
    $("currentSession").textContent = data.session_id;
    renderMemories("memoryList", data.retrieved_memories);
    await refreshSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
  } finally {
    $("sendButton").disabled = false;
    input.focus();
  }
}

async function finalizeCurrent() {
  if (!state.currentSessionId) return;
  await api(`/api/sessions/${encodeURIComponent(state.currentSessionId)}/finalize`, { method: "POST" });
  await refreshSessions();
}

async function buildMemory() {
  const body = {
    session_id: state.currentSessionId,
    sample_id: "demo_user",
    build_hierarchical: true,
    build_episodic: true,
    build_entity_relation: true,
  };
  const data = await api("/api/memory/build", { method: "POST", body: JSON.stringify(body) });
  $("buildStatus").textContent = JSON.stringify(data, null, 2);
}

async function searchMemory() {
  const query = $("searchInput").value.trim();
  if (!query) return;
  const data = await api("/api/memory/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: 5 }),
  });
  renderMemories("searchResults", data.results);
}

async function resetDemo() {
  await api("/api/reset", { method: "POST" });
  state.currentSessionId = null;
  $("currentSession").textContent = "none";
  $("chatLog").innerHTML = "";
  renderMemories("memoryList", []);
  renderMemories("searchResults", []);
  $("buildStatus").textContent = "Not started.";
  await refreshHealth();
  await refreshSessions();
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

$("chatForm").addEventListener("submit", sendMessage);
$("refreshSessions").addEventListener("click", refreshSessions);
$("finalizeSession").addEventListener("click", finalizeCurrent);
$("buildMemory").addEventListener("click", buildMemory);
$("searchButton").addEventListener("click", searchMemory);
$("resetDemo").addEventListener("click", resetDemo);
$("messageInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chatForm").requestSubmit();
  }
});

refreshHealth().then(refreshSessions).catch((error) => setStatus(`Backend error: ${error.message}`));
