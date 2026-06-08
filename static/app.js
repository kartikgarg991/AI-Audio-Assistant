const state = {
  sessionId: crypto.randomUUID(),
  audioId: null,
  recorder: null,
  recordingChunks: [],
  recordingBlob: null,
  timer: null,
  startedAt: null,
  logPoller: null,
};

const $ = (selector) => document.querySelector(selector);
const progressBox = $("#progressBox");
const errorBox = $("#errorBox");

function languageMode() {
  return document.querySelector('input[name="language"]:checked').value;
}

function setBusy(busy, title = "Processing audio", text = "This can take a little while on free services.") {
  progressBox.classList.toggle("hidden", !busy);
  $("#progressTitle").textContent = title;
  $("#progressText").textContent = text;
  document.querySelectorAll("button").forEach((button) => {
    if (!["recordButton", "newSessionButton"].includes(button.id)) button.disabled = busy;
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(value) {
  const lines = escapeHtml(value).split(/\r?\n/);
  const html = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };

  const renderInline = (text) => text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${renderInline(bullet[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${renderInline(trimmed)}</p>`);
  }

  closeList();
  return html.join("");
}

function renderLogs(logs = []) {
  $("#logPanel").classList.toggle("hidden", logs.length === 0);
  $("#logCount").textContent = `${logs.length} event${logs.length === 1 ? "" : "s"}`;
  $("#logList").innerHTML = logs.map((entry) => {
    const time = new Date((entry.ts || 0) * 1000).toLocaleTimeString();
    const level = entry.level || "info";
    return `<div class="log-line ${escapeHtml(level)}">[${escapeHtml(time)}] ${escapeHtml(level.toUpperCase())} ${escapeHtml(entry.message)}</div>`;
  }).join("");
  $("#logList").scrollTop = $("#logList").scrollHeight;
}

async function refreshLogs() {
  try {
    const data = await api(`/api/session/${state.sessionId}/logs`);
    renderLogs(data.logs || []);
  } catch {
    // Logs are a debugging aid; avoid interrupting the main workflow.
  }
}

function startLogPolling() {
  clearInterval(state.logPoller);
  renderLogs([]);
  refreshLogs();
  state.logPoller = setInterval(refreshLogs, 1000);
}

function stopLogPolling() {
  clearInterval(state.logPoller);
  state.logPoller = null;
  refreshLogs();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.classList.add("hidden");
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || "Request failed.");
  return data;
}

function showTranscript(data) {
  state.audioId = data.audio_id;
  $("#transcriptText").value = data.text || "";
  $("#providerLabel").textContent = `Transcribed with ${data.provider}${data.fallback_used ? " (fallback used)" : ""}. You can edit it before indexing.`;
  $("#createChatButton").disabled = false;
  $("#transcriptCard").classList.remove("hidden");
  $("#transcriptCard").scrollIntoView({ behavior: "smooth" });
}

async function uploadAudio(file, sourceType) {
  clearError();
  if (!file) return showError("Choose or record an audio file first.");
  const form = new FormData();
  form.append("file", file, file.name || "recording.webm");
  form.append("session_id", state.sessionId);
  form.append("language_mode", languageMode());
  form.append("source_type", sourceType);
  setBusy(true, "Transcribing audio", "Audio is being normalized, chunked, and sent to the selected provider.");
  startLogPolling();
  try {
    showTranscript(await api("/api/transcribe/upload", { method: "POST", body: form }));
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    stopLogPolling();
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");
    $(`#${tab.dataset.tab}Panel`).classList.add("active");
  });
});

$("#youtubeButton").addEventListener("click", async () => {
  clearError();
  const url = $("#youtubeUrl").value.trim();
  if (!url) return showError("Enter a YouTube URL.");
  setBusy(true, "Downloading and transcribing", "YouTube extraction can be slower on free hosting.");
  startLogPolling();
  try {
    showTranscript(await api("/api/transcribe/youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, url, language_mode: languageMode() }),
    }));
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    stopLogPolling();
  }
});

$("#uploadButton").addEventListener("click", () => uploadAudio($("#audioFile").files[0], "upload"));

$("#recordButton").addEventListener("click", async () => {
  if (state.recorder?.state === "recording") {
    state.recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.recordingChunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (event) => event.data.size && state.recordingChunks.push(event.data);
    state.recorder.onstop = () => {
      state.recordingBlob = new Blob(state.recordingChunks, { type: state.recorder.mimeType || "audio/webm" });
      stream.getTracks().forEach((track) => track.stop());
      clearInterval(state.timer);
      $("#recordDot").classList.remove("live");
      $("#recordStatus").textContent = "Recording ready";
      $("#recordButton").textContent = "Record again";
      $("#recordProcessButton").disabled = false;
    };
    state.recorder.start();
    state.startedAt = Date.now();
    $("#recordDot").classList.add("live");
    $("#recordStatus").textContent = "Recording";
    $("#recordButton").textContent = "Stop recording";
    state.timer = setInterval(() => {
      const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
      $("#recordTimer").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }, 500);
  } catch {
    showError("Microphone permission was denied or is unavailable.");
  }
});

$("#recordProcessButton").addEventListener("click", () => {
  if (!state.recordingBlob) return;
  const file = new File([state.recordingBlob], "recording.webm", { type: state.recordingBlob.type });
  uploadAudio(file, "microphone");
});

$("#saveButton").addEventListener("click", async () => {
  try {
    await api("/api/transcript", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        audio_id: state.audioId,
        transcript: $("#transcriptText").value,
      }),
    });
    $("#saveStatus").textContent = "Changes saved.";
    $("#createChatButton").disabled = false;
  } catch (error) {
    showError(error.message);
  }
});

$("#transcriptText").addEventListener("input", () => {
  $("#saveStatus").textContent = "Unsaved transcript changes.";
  $("#createChatButton").disabled = true;
});

$("#createChatButton").addEventListener("click", async () => {
  clearError();
  setBusy(true, "Creating your audio workspace", "Generating title, summary, embeddings, and the Pinecone index.");
  startLogPolling();
  try {
    const data = await api("/api/chat/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        audio_id: state.audioId,
        transcript: $("#transcriptText").value,
      }),
    });
    $("#resultTitle").textContent = data.title;
    $("#summaryText").textContent = data.summary;
    $("#resultCard").classList.remove("hidden");
    $("#resultCard").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    stopLogPolling();
  }
});

function addMessage(text, role) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  if (role === "assistant") {
    message.innerHTML = renderMarkdown(text);
  } else {
    message.textContent = text;
  }
  $("#chatMessages").appendChild(message);
  message.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#questionInput");
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, "user");
  input.value = "";
  try {
    const data = await api("/api/chat/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, audio_id: state.audioId, question }),
    });
    addMessage(data.answer, "assistant");
  } catch (error) {
    addMessage(error.message, "assistant");
  }
});

$("#newSessionButton").addEventListener("click", async () => {
  try {
    await fetch(`/api/session/${state.sessionId}`, { method: "DELETE" });
  } finally {
    location.reload();
  }
});

api("/api/session/touch", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id: state.sessionId }),
}).catch(() => {});
