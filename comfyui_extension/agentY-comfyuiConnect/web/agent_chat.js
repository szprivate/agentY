import { app } from "../../scripts/app.js";

// agentY chat — a ComfyUI sidebar tab that talks to the agentY headless chat host
// (src/utils/agentY_server.py on :5000) over HTTP/SSE. It replaces the Chainlit
// GUI: the agent's *text* streams into this panel, while every generated image /
// video is dropped onto the ComfyUI graph as a LoadImage / video-loader node
// (see onOutput → injectNode). Conversations, slash commands, and thread history
// mirror what the old Chainlit UI offered.

const DEFAULT_PORT = 5000;

function backendBase() {
  return (
    localStorage.getItem("agentY_backend") ||
    `http://${location.hostname || "127.0.0.1"}:${DEFAULT_PORT}`
  );
}

// ── tiny helpers ──────────────────────────────────────────────────────────────
function el(tag, props = {}, children = []) {
  const n = document.createElement(tag);
  Object.assign(n, props);
  if (props.style) Object.assign(n.style, props.style);
  for (const c of [].concat(children)) n.append(c);
  return n;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
// Minimal markdown: **bold**, `code`, newlines. Enough for the agent's messages.
function mdToHtml(s) {
  let h = escapeHtml(s);
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\n/g, "<br>");
  return h;
}

const SLASH_FALLBACK = [
  { name: "/restart", description: "Restart the agent pipeline" },
  { name: "/stop", description: "Stop and shut down the agent" },
  { name: "/unload", description: "Unload Ollama models from VRAM" },
  { name: "/clear_vram", description: "Clear ComfyUI GPU VRAM" },
  { name: "/images", description: "List images generated in this thread" },
  { name: "/clearhistory", description: "Delete all conversation history" },
  { name: "/switch_model", description: "Switch an agent's LLM" },
  { name: "/add_workflow", description: "Add a workflow (JSON path, or 'canvas <name>' for the open graph)" },
  { name: "/resend", description: "Resend the first user message" },
  { name: "/remove_workflow", description: "Remove a workflow by name" },
];

// Model presets for the quick-switch dropdown (grouped by provider). Each entry
// is [ "<provider>,<model>", "Display name" ] — the provider,model string is
// exactly what /switch_model expects. Edit this list to add your own models.
const MODEL_PRESETS = {
  Anthropic: [
    ["claude,claude-haiku-4-5", "Claude Haiku 4.5"],
    ["claude,claude-sonnet-4-5", "Claude Sonnet 4.5"],
  ],
  "Alibaba (DashScope)": [
    ["dashscope,qwen-plus", "Qwen Plus"],
    ["dashscope,qwen3.7-plus", "Qwen3.7 Plus"],
    ["dashscope,qwen-max", "Qwen Max"],
  ],
};

// Which agent(s) the model switch targets.
const MODEL_TARGETS = [
  ["all", "All agents"],
  ["orchestrator", "Orchestrator"],
];

class AgentChat {
  constructor(root) {
    this.root = root;
    this.threadId = null;
    this.streaming = false;
    this.activeAsk = null; // request_id awaiting a reply
    this.curRequestId = null; // request_id of the in-flight turn (for Stop)
    this.abortController = null; // aborts the SSE fetch on Stop
    this._stopping = false; // set while a user-initiated stop is in progress
    this.attachments = []; // [{path,name}]
    this.commands = SLASH_FALLBACK;
    this.curAssistant = null; // DOM node currently streaming assistant text
    this.curStep = null; // {details, body}
    this.nodeCount = 0;
    this._selOrder = []; // node ids in the order they were selected on the canvas
    this._consumed = {}; // nodeId -> value already sent as an input (skip re-sending unchanged)
    this.domCache = new Map(); // threadId -> {html, scroll}: live-rendered panel (thinking/step blocks) kept across conversation switches
    this._injectStyles();
    this._build();
    this._loadCommands();
    this._loadModels();
    this._loadThreads();
    this.newThread();
  }

  // ── styling ────────────────────────────────────────────────────────────────
  _injectStyles() {
    if (document.getElementById("agentY-chat-styles")) return;
    const css = `
    .ay-wrap{
      --ay-bg:#262624; --ay-surface:#302f2c; --ay-surface2:#3b3936;
      --ay-border:rgba(240,235,225,.10); --ay-text:#f2f0ea; --ay-muted:#a8a39a;
      --ay-accent:#d97757; --ay-accent2:#c56647; --ay-accent-soft:rgba(217,119,87,.15);
      display:flex;flex-direction:column;height:100%;
      font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
      font-size:13.5px;line-height:1.5;color:var(--ay-text);background:var(--ay-bg);
    }
    .ay-bar{display:flex;gap:8px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--ay-border);flex-shrink:0;}
    .ay-bar select{flex:1;background:var(--ay-surface);color:var(--ay-text);border:1px solid var(--ay-border);border-radius:10px;padding:7px 10px;font-size:12.5px;cursor:pointer;}
    .ay-btn{background:var(--ay-surface2);color:var(--ay-text);border:1px solid var(--ay-border);border-radius:10px;padding:7px 11px;cursor:pointer;font-size:12.5px;transition:background .12s,border-color .12s,transform .06s;}
    .ay-btn:hover{background:#464440;}
    .ay-btn:active{transform:translateY(1px);}
    .ay-btn.ay-send{background:var(--ay-accent);color:#2a1810;border-color:transparent;border-radius:999px;padding:9px 18px;font-weight:600;}
    .ay-btn.ay-send:hover{background:var(--ay-accent2);}
    .ay-btn.ay-stop{background:#8a4034;color:#ffe1d9;border-color:transparent;border-radius:999px;}
    .ay-btn.ay-stop:hover{background:#9c4a3c;}
    .ay-log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;}
    .ay-msg{padding:10px 13px;border-radius:16px;max-width:92%;word-wrap:break-word;line-height:1.5;}
    .ay-user{background:var(--ay-accent-soft);border:1px solid rgba(217,119,87,.28);align-self:flex-end;border-bottom-right-radius:5px;}
    .ay-assistant{background:var(--ay-surface);align-self:flex-start;border-bottom-left-radius:5px;}
    .ay-system{background:transparent;color:var(--ay-muted);font-size:12px;align-self:center;text-align:center;max-width:100%;padding:2px 8px;}
    .ay-ask{background:rgba(217,119,87,.10);color:#f0d9c2;border:1px solid rgba(217,119,87,.35);align-self:stretch;max-width:100%;}
    .ay-code{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,monospace;background:rgba(0,0,0,.25);padding:2px 5px;border-radius:6px;font-size:12px;}
    .ay-step{border:1px solid var(--ay-border);border-radius:12px;background:var(--ay-surface);overflow:hidden;align-self:stretch;}
    .ay-step>summary{cursor:pointer;padding:8px 12px;color:var(--ay-muted);font-weight:600;font-size:12px;list-style:none;}
    .ay-step>summary::-webkit-details-marker{display:none;}
    .ay-step>summary::before{content:"▸ ";opacity:.7;}
    .ay-step[open]>summary::before{content:"▾ ";}
    .ay-step .ay-step-body{padding:8px 12px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,monospace;font-size:11px;color:var(--ay-muted);max-height:240px;overflow:auto;word-break:break-word;border-top:1px solid var(--ay-border);}
    .ay-step.ay-tool{border-color:rgba(127,212,160,.22);}
    .ay-step.ay-tool>summary{color:#8fd6ab;}
    .ay-status{font-size:11px;color:var(--ay-muted);padding:2px 12px;font-family:ui-monospace,monospace;align-self:center;}
    .ay-inwrap{border-top:1px solid var(--ay-border);padding:10px 12px;display:flex;flex-direction:column;gap:8px;flex-shrink:0;position:relative;background:var(--ay-bg);}
    .ay-attach{display:flex;flex-wrap:wrap;gap:5px;}
    .ay-chip{background:var(--ay-surface2);border:1px solid var(--ay-border);border-radius:999px;padding:3px 9px;font-size:11px;color:var(--ay-text);}
    .ay-inrow{display:flex;gap:8px;align-items:flex-end;}
    .ay-input{flex:1;resize:none;min-height:40px;max-height:150px;background:var(--ay-surface);color:var(--ay-text);border:1px solid var(--ay-border);border-radius:14px;padding:10px 13px;font-family:inherit;font-size:13.5px;line-height:1.5;outline:none;transition:border-color .12s;}
    .ay-input:focus{border-color:rgba(217,119,87,.55);}
    .ay-input::placeholder{color:var(--ay-muted);}
    .ay-modelbar{display:flex;align-items:center;gap:7px;padding:8px 12px 10px;border-top:1px solid var(--ay-border);flex-shrink:0;background:var(--ay-bg);}
    .ay-mlabel{color:var(--ay-muted);font-size:11.5px;flex-shrink:0;}
    .ay-modelbar select{background:var(--ay-surface);color:var(--ay-text);border:1px solid var(--ay-border);border-radius:9px;padding:6px 9px;font-size:12px;cursor:pointer;transition:border-color .12s;}
    .ay-modelbar select:hover{border-color:rgba(217,119,87,.45);}
    .ay-modelbar select:disabled{opacity:.45;cursor:not-allowed;}
    .ay-mmodel{flex:1;min-width:0;}
    .ay-pop{position:absolute;bottom:100%;left:12px;right:12px;margin-bottom:6px;background:var(--ay-surface);border:1px solid var(--ay-border);border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.5);z-index:50;max-height:280px;overflow:auto;display:none;}
    .ay-pop-item{padding:8px 12px;cursor:pointer;display:flex;gap:10px;align-items:baseline;}
    .ay-pop-item:hover,.ay-pop-item.sel{background:var(--ay-surface2);}
    .ay-pop-item.sel{box-shadow:inset 3px 0 0 var(--ay-accent);}
    .ay-pop-name{font-family:ui-monospace,monospace;color:var(--ay-accent);min-width:130px;font-size:12.5px;}
    .ay-pop-desc{color:var(--ay-muted);font-size:12px;}
    .ay-log::-webkit-scrollbar,.ay-step-body::-webkit-scrollbar,.ay-pop::-webkit-scrollbar{width:8px;height:8px;}
    .ay-log::-webkit-scrollbar-thumb,.ay-step-body::-webkit-scrollbar-thumb,.ay-pop::-webkit-scrollbar-thumb{background:var(--ay-surface2);border-radius:8px;}
    `;
    document.head.append(el("style", { id: "agentY-chat-styles", textContent: css }));
  }

  // ── DOM ────────────────────────────────────────────────────────────────────
  _build() {
    this.root.innerHTML = "";
    const wrap = el("div", { className: "ay-wrap" });

    // top bar: thread selector + new + delete
    this.threadSel = el("select", { title: "Conversation" });
    this.threadSel.addEventListener("change", () => this.openThread(this.threadSel.value));
    const newBtn = el("button", { className: "ay-btn", textContent: "+ New", title: "New chat" });
    newBtn.addEventListener("click", () => this.newThread());
    const delBtn = el("button", { className: "ay-btn", textContent: "🗑", title: "Delete this conversation" });
    delBtn.addEventListener("click", () => this.deleteThread());
    wrap.append(el("div", { className: "ay-bar" }, [this.threadSel, newBtn, delBtn]));

    // message log
    this.logEl = el("div", { className: "ay-log" });
    wrap.append(this.logEl);

    // input area
    this.attachEl = el("div", { className: "ay-attach" });
    this.pop = el("div", { className: "ay-pop" });
    this.input = el("textarea", { className: "ay-input", placeholder: "Message agentY…  (type / for commands)" });
    this.input.addEventListener("input", () => this._onInput());
    this.input.addEventListener("keydown", (e) => this._onKeydown(e));

    const attachBtn = el("button", { className: "ay-btn", textContent: "📎", title: "Attach image" });
    this.fileInput = el("input", { type: "file", accept: "image/*", multiple: true, style: { display: "none" } });
    this.fileInput.addEventListener("change", () => this._onFiles());
    attachBtn.addEventListener("click", () => this.fileInput.click());

    this.sendBtn = el("button", { className: "ay-btn ay-send", textContent: "Send" });
    this.sendBtn.addEventListener("click", () => this._onSendBtn());

    const inrow = el("div", { className: "ay-inrow" }, [attachBtn, this.input, this.sendBtn]);
    const inwrap = el("div", { className: "ay-inwrap" }, [this.pop, this.attachEl, inrow, this.fileInput]);
    wrap.append(inwrap);

    // model quick-switch bar (bottom)
    wrap.append(this._buildModelBar());

    this.root.append(wrap);
  }

  // ── model quick-switch bar ───────────────────────────────────────────────────
  _buildModelBar() {
    this.targetSel = el("select", { className: "ay-mtarget", title: "Which agent(s) to switch" });
    for (const [val, label] of MODEL_TARGETS) {
      this.targetSel.append(el("option", { value: val, textContent: label }));
    }
    this.modelSel = el("select", { className: "ay-mmodel", title: "Switch model" });
    // Seed with the static presets; _loadModels() replaces this at startup with
    // the vendors/models actually available (Ollama installed list, and
    // Anthropic/DashScope only when their API key is set).
    this._populateModelSelect(MODEL_PRESETS);
    this.modelSel.addEventListener("change", () => this._applyModel());
    return el("div", { className: "ay-modelbar" }, [
      el("span", { className: "ay-mlabel", textContent: "Model" }),
      this.modelSel,
      this.targetSel,
    ]);
  }

  // Rebuild the model dropdown from a { "<vendor>": [[spec,label],…] } map,
  // preserving the current selection where possible.
  _populateModelSelect(groups) {
    const sel = this.modelSel;
    const cur = sel.value;
    sel.innerHTML = "";
    sel.append(el("option", { value: "", textContent: "🔀 Switch model…" }));
    for (const [group, models] of Object.entries(groups || {})) {
      if (!models || !models.length) continue;
      const og = el("optgroup", { label: group });
      for (const [spec, label] of models) og.append(el("option", { value: spec, textContent: label }));
      sel.append(og);
    }
    if (cur) sel.value = cur;
  }

  // Fetch the live vendor/model list from the host; fall back to static presets.
  async _loadModels() {
    try {
      const r = await fetch(backendBase() + "/agentY/models");
      if (r.ok) {
        const groups = await r.json();
        if (groups && Object.keys(groups).length) { this._populateModelSelect(groups); return; }
      }
    } catch (_) {}
    this._populateModelSelect(MODEL_PRESETS);
  }

  async _applyModel() {
    const spec = this.modelSel.value;
    if (!spec) return;
    if (this.streaming) { this.modelSel.value = ""; return; } // don't switch mid-turn
    const target = this.targetSel.value || "all";
    this.modelSel.value = ""; // reset to placeholder
    try {
      const r = await fetch(backendBase() + "/agentY/switch_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target, spec }),
      });
      const j = await r.json();
      (j.messages && j.messages.length ? j.messages : [(j.ok ? "✅ Model switched." : "❌ Switch failed.")])
        .forEach((m) => this._sys(m));
    } catch (e) {
      this._sys("❌ Switch failed: " + e);
    }
    this._savePanel();
  }

  // ── backend calls ───────────────────────────────────────────────────────────
  async _loadCommands() {
    try {
      const r = await fetch(backendBase() + "/agentY/commands");
      if (r.ok) this.commands = await r.json();
    } catch (_) {}
  }

  async _loadThreads() {
    try {
      const r = await fetch(backendBase() + "/agentY/threads");
      const list = r.ok ? await r.json() : [];
      const cur = this.threadId;
      this.threadSel.innerHTML = "";
      for (const t of list) {
        this.threadSel.append(el("option", { value: t.id, textContent: t.title || "New chat" }));
      }
      if (cur) this.threadSel.value = cur;
    } catch (_) {}
  }

  // Snapshot the current thread's live-rendered panel (thinking/step blocks and
  // all) so returning to it later this session restores exactly what was shown.
  _saveCurrentDom() {
    if (this.threadId) {
      this.domCache.set(this.threadId, { html: this.logEl.innerHTML, scroll: this.logEl.scrollTop });
    }
  }

  // Persist the rendered panel (collapsible think/step blocks and all) to the
  // backend so it survives page reloads / new sessions, not just in-session
  // switches. Fire-and-forget.
  _savePanel() {
    if (!this.threadId) return;
    fetch(backendBase() + "/agentY/threads/" + this.threadId + "/panel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ html: this.logEl.innerHTML }),
    }).catch(() => {});
  }

  newThread() {
    this._saveCurrentDom();
    this.threadId = null;
    this.logEl.innerHTML = "";
    this._sys("New conversation. Ask me to generate or edit an image/video — results drop onto the graph as nodes.");
  }

  async deleteThread() {
    if (!this.threadId) return this.newThread();
    const tid = this.threadId;
    try {
      await fetch(backendBase() + "/agentY/threads/" + tid, { method: "DELETE" });
    } catch (_) {}
    this.domCache.delete(tid);
    this.threadId = null; // so newThread() doesn't re-cache the just-deleted thread
    this.newThread();
    this._loadThreads();
  }

  async openThread(id) {
    if (!id || id === this.threadId) return;
    this._saveCurrentDom();
    this.threadId = id;
    // Restore the live-rendered panel if we've shown this thread already this
    // session (keeps the thinking/step blocks); otherwise rebuild from the
    // persisted messages, which store only the final user/assistant text.
    const cached = this.domCache.get(id);
    if (cached) {
      this.logEl.innerHTML = cached.html;
      this.logEl.scrollTop = cached.scroll;
      return;
    }
    this.logEl.innerHTML = "";
    try {
      const r = await fetch(backendBase() + "/agentY/threads/" + id);
      if (!r.ok) return;
      const t = await r.json();
      // Prefer the persisted rendered panel — collapsible think/step blocks
      // intact, survives page reloads — and only fall back to the text-only
      // message log for threads that were never rendered (e.g. pre-dating this).
      if (t.panel_html) {
        this.logEl.innerHTML = t.panel_html;
        this.logEl.scrollTop = this.logEl.scrollHeight;
        return;
      }
      for (const m of t.messages || []) {
        if (m.role === "user") this._userMsg(m.content);
        else if (m.role === "assistant") this._assistantMsg(m.content);
        else this._sys(m.content);
      }
    } catch (_) {}
  }

  // ── rendering ────────────────────────────────────────────────────────────────
  _scroll() { this.logEl.scrollTop = this.logEl.scrollHeight; }
  _sys(text) {
    this.logEl.append(el("div", { className: "ay-msg ay-system", innerHTML: mdToHtml(text) }));
    this._scroll();
  }
  _userMsg(text) {
    this.logEl.append(el("div", { className: "ay-msg ay-user", innerHTML: mdToHtml(text) }));
    this._scroll();
  }
  _assistantMsg(text) {
    this.logEl.append(el("div", { className: "ay-msg ay-assistant", innerHTML: mdToHtml(text) }));
    this._scroll();
  }
  _ensureAssistant() {
    if (!this.curAssistant) {
      this.curAssistant = el("div", { className: "ay-msg ay-assistant" });
      this.curAssistant._raw = "";
      this.logEl.append(this.curAssistant);
    }
    return this.curAssistant;
  }
  _appendAssistant(text) {
    const m = this._ensureAssistant();
    m._raw += text;
    m.innerHTML = mdToHtml(m._raw);
    this._scroll();
  }
  _stepStart(name) {
    const details = el("details", { className: "ay-step", open: false });
    const body = el("div", { className: "ay-step-body" });
    details.append(el("summary", { textContent: name }), body);
    this.logEl.append(details);
    this.curStep = { details, body, name };
    this._scroll();
  }
  _stepText(text) {
    if (!this.curStep) this._stepStart("working");
    this.curStep.body.textContent += text;
    this._scroll();
  }
  _stepEnd() { this.curStep = null; }
  // Render an agent tool call / result as a collapsible block, inline in the
  // chat log (so it persists via _savePanel like every other block).
  _toolBlock(ev) {
    this.curAssistant = null; // close the current text bubble; keep ordering
    this._toolBlocks = this._toolBlocks || {};
    const id = ev.id || "";
    if (ev.phase === "call") {
      const details = el("details", { className: "ay-step ay-tool", open: false });
      const body = el("div", { className: "ay-step-body" });
      body.textContent = ev.input ? "input: " + ev.input : "(no input)";
      details.append(el("summary", { textContent: "🔧 " + (ev.name || "tool") }), body);
      this.logEl.append(details);
      if (id) this._toolBlocks[id] = { details, body };
    } else {
      const blk = id && this._toolBlocks[id];
      if (blk) {
        blk.body.textContent += "\n\n→ " + (ev.result || "(done)");
      } else {
        const details = el("details", { className: "ay-step ay-tool", open: false });
        details.append(
          el("summary", { textContent: "🔧 " + (ev.name || "tool") }),
          el("div", { className: "ay-step-body", textContent: "→ " + (ev.result || "(done)") }),
        );
        this.logEl.append(details);
      }
    }
    this._scroll();
  }
  _status(text) {
    if (!this._statusEl || !this._statusEl.isConnected) {
      this._statusEl = el("div", { className: "ay-status" });
      this.logEl.append(this._statusEl);
    }
    this._statusEl.textContent = text;
    this._scroll();
  }
  _clearStatus() { this._statusEl = null; }

  // ── graph node injection (the whole point) ───────────────────────────────────
  injectNode(ev) {
    const LG = window.LiteGraph;
    const cands = ev.node_candidates || (ev.kind === "video" ? ["VHS_LoadVideo", "LoadVideo"] : ["LoadImage"]);
    const type = cands.find((t) => LG && LG.registered_node_types && LG.registered_node_types[t]);
    if (!type) {
      this._sys(`⚠️ ${ev.kind} saved at \`${ev.path}\` — no loader node available in this ComfyUI. Load it manually.`);
      return;
    }
    let node;
    try {
      node = LG.createNode(type);
      app.graph.add(node);
    } catch (e) {
      this._sys(`⚠️ Could not add ${type} node: ${e}`);
      return;
    }
    // Stagger positions near the canvas so multiple outputs don't stack exactly.
    const off = this.nodeCount++ * 40;
    node.pos = [80 + off, 80 + off];
    const val = ev.filename || ev.path;
    const wnames = ev.kind === "image" ? ["image"] : ["video", "file", "path"];
    const w = (node.widgets || []).find((x) => wnames.includes(x.name));
    if (w) {
      if (w.options && Array.isArray(w.options.values) && ev.filename && !w.options.values.includes(ev.filename)) {
        w.options.values.push(ev.filename);
      }
      w.value = val;
      try { if (w.callback) w.callback(val); } catch (_) {}
    }
    node.title = "agentY · " + (ev.name || type);
    app.graph.setDirtyCanvas(true, true);
    this._sys(`🧩 Added **${type}** node → \`${ev.name}\``);
  }

  // ── SSE event dispatch ───────────────────────────────────────────────────────
  _onEvent(ev) {
    switch (ev.type) {
      case "thread":
        if (ev.id && ev.id !== this.threadId) { this.threadId = ev.id; this._loadThreads(); }
        break;
      case "request":
        this.curRequestId = ev.request_id;
        break;
      case "text":
        this._appendAssistant(ev.data);
        break;
      case "think":
        // fold reasoning into a collapsible thinking step, inline in the chat
        this.curAssistant = null;
        if (!this._thinkStep || !this._thinkStep.details.isConnected) {
          this._stepStart("💭 thinking");
          this._thinkStep = this.curStep;
          this.curStep = null;
        }
        this._thinkStep.body.textContent += ev.data;
        this._scroll();
        break;
      case "tool":
        // render the agent's tool call / result as an inline collapsible block
        this._toolBlock(ev);
        break;
      case "step_start":
        this.curAssistant = null;
        this._stepStart(ev.name || "step");
        break;
      case "step_text":
        this._stepText(ev.data);
        break;
      case "step_end":
        this._thinkStep = null;
        this._stepEnd();
        break;
      case "progress":
      case "qa":
        this._status(ev.data);
        break;
      case "exec":
        if (ev.state === "start") this._status("⚙️ ComfyUI running…");
        else this._clearStatus();
        break;
      case "plan":
        this._sys("🗂️ **Plan:**\n" + (ev.steps || []).map((s, i) => `${i + 1}. ${s}`).join("\n"));
        break;
      case "plan_step":
        break; // (kept lightweight)
      case "output":
        this.curAssistant = null;
        this.injectNode(ev);
        break;
      case "canvas_patch":
        this.curAssistant = null;
        this._applyCanvasPatch(ev);
        break;
      case "system":
        this.curAssistant = null;
        this._sys(ev.data);
        break;
      case "ask":
        this.curAssistant = null;
        this.activeAsk = ev.request_id;
        this._setBusy(true); // awaiting a reply → button reverts to Send
        this.logEl.append(el("div", { className: "ay-msg ay-ask", innerHTML: mdToHtml("⏸️ " + ev.prompt) }));
        this._scroll();
        this.input.focus();
        break;
      case "error":
        this.curAssistant = null;
        this._sys("❌ " + ev.message);
        break;
      case "done":
        this._clearStatus();
        this.curStep = null;
        this.curAssistant = null;
        this._thinkStep = null;
        this._toolBlocks = {};
        this.streaming = false;
        this._setBusy(false);
        this._savePanel();  // persist the rendered panel so blocks survive reloads
        this._loadThreads();
        break;
    }
  }

  async _stream(body) {
    this.streaming = true;
    this._stopping = false;
    this._thinkStep = null;
    this._toolBlocks = {};
    this.abortController = new AbortController();
    this._setBusy(true);
    try {
      const resp = await fetch(backendBase() + "/agentY/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: this.abortController.signal,
      });
      if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const line = frame.split("\n").find((l) => l.startsWith("data:"));
          if (line) {
            try { this._onEvent(JSON.parse(line.slice(line.indexOf(":") + 1).trim())); }
            catch (e) { console.error("[agentY] bad SSE frame", e); }
          }
        }
      }
    } catch (e) {
      // A user-initiated Stop aborts the fetch → don't show it as an error.
      if (!this._stopping && e.name !== "AbortError") {
        this._sys("❌ Connection error: " + e + `\n\nIs the agentY chat host running? (\`run_agent.ps1\`, ${backendBase()})`);
      }
    } finally {
      this.streaming = false;
      this.abortController = null;
      this._setBusy(false);
    }
  }

  // Button doubles as Send / Stop depending on state.
  _onSendBtn() {
    if (this.streaming && !this.activeAsk) this._stop();
    else this.send();
  }

  async _stop() {
    if (!this.streaming) return;
    this._stopping = true;
    this._status("⏹ Stopping…");
    // Ask the backend to cancel the run (halts the agent loop + interrupts ComfyUI).
    try {
      await fetch(backendBase() + "/agentY/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: this.curRequestId, thread_id: this.threadId }),
      });
    } catch (_) {}
    // Stop consuming the SSE stream client-side.
    try { if (this.abortController) this.abortController.abort(); } catch (_) {}
    this._clearStatus();
    this._sys("⏹ Stopped.");
    this.curAssistant = null;
    this.curStep = null;
    this.streaming = false;
    this._setBusy(false);
    this._savePanel();
  }

  _setBusy(b) {
    // While a turn is running (and not waiting on a reply) the button becomes a
    // Stop button; otherwise it's the Send/reply button. Always clickable.
    const stopMode = b && !this.activeAsk;
    this.sendBtn.disabled = false;
    this.sendBtn.textContent = stopMode ? "⏹ Stop" : "Send";
    this.sendBtn.classList.toggle("ay-stop", stopMode);
    // Don't allow a model switch mid-turn.
    if (this.modelSel) this.modelSel.disabled = b;
    if (this.targetSel) this.targetSel.disabled = b;
  }

  // ── sending ──────────────────────────────────────────────────────────────────
  async send() {
    const text = this.input.value.trim();
    const canvasInputs = this._collectCanvasInputs();
    const canvasHooks = this._collectCanvasHooks();
    const canvasSelection = this._collectCanvasSelection();
    if (!text && this.attachments.length === 0 && canvasInputs.length === 0 &&
        canvasHooks.length === 0) return;

    // Answering an interactive ask → side-channel reply; the SSE stream continues.
    if (this.activeAsk) {
      const rid = this.activeAsk;
      this.activeAsk = null;
      this._setBusy(this.streaming); // reply sent → back to Stop while it continues
      this._userMsg(text || "(continue)");
      this.input.value = "";
      this._autosize();
      try {
        await fetch(backendBase() + "/agentY/reply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: rid, text }),
        });
      } catch (e) { this._sys("❌ Reply failed: " + e); }
      return;
    }

    if (this.streaming) return; // one turn at a time
    const imgs = this.attachments.map((a) => a.path);
    const noteParts = [];
    if (this.attachments.length) noteParts.push(`${this.attachments.length} image(s) attached`);
    if (canvasInputs.length) {
      const ni = canvasInputs.filter((c) => c.kind === "image").length;
      const nv = canvasInputs.length - ni;
      const bits = [];
      if (ni) bits.push(`${ni} image`);
      if (nv) bits.push(`${nv} video`);
      noteParts.push(`${bits.join(" + ")} from canvas`);
    }
    if (canvasHooks.length) noteParts.push(`${canvasHooks.length} canvas hook(s)`);
    this._userMsg(text + (noteParts.length ? `  \n_(${noteParts.join(", ")})_` : ""));
    this.input.value = "";
    this._autosize();
    this.attachments = [];
    this._renderAttachments();
    this._hidePop();
    // Mark these canvas files as consumed so an unchanged, still-selected node
    // isn't re-sent on the next message.
    for (const ci of canvasInputs) if (ci._nodeId != null) this._consumed[ci._nodeId] = ci.value;
    // Always capture the on-canvas graph as an API prompt so the agent can act
    // on it — hooks drive the "run my canvas graph" path, and "add the workflow
    // open in the canvas" (chat or /add_workflow canvas <name>) needs it too.
    // graphToPrompt() is what ComfyUI runs on every Queue, so the cost is negligible.
    const canvasPrompt = await this._captureCanvasGraph();
    await this._stream({
      thread_id: this.threadId,
      message: text,
      image_paths: imgs,
      canvas_inputs: canvasInputs.map((c) => ({ value: c.value, kind: c.kind })),
      canvas_hooks: canvasHooks,
      canvas_selection: canvasSelection,
      canvas_prompt: canvasPrompt,
    });
  }

  // ── attachments ──────────────────────────────────────────────────────────────
  async _onFiles() {
    for (const f of this.fileInput.files) {
      const fd = new FormData();
      fd.append("file", f);
      try {
        const r = await fetch(backendBase() + "/agentY/upload", { method: "POST", body: fd });
        if (r.ok) { const j = await r.json(); this.attachments.push({ path: j.path, name: j.name }); }
      } catch (e) { this._sys("❌ Upload failed: " + e); }
    }
    this.fileInput.value = "";
    this._renderAttachments();
  }
  _renderAttachments() {
    this.attachEl.innerHTML = "";
    this.attachments.forEach((a, i) => {
      const chip = el("span", { className: "ay-chip", textContent: "📎 " + a.name + "  ✕" });
      chip.style.cursor = "pointer";
      chip.title = "remove";
      chip.addEventListener("click", () => { this.attachments.splice(i, 1); this._renderAttachments(); });
      this.attachEl.append(chip);
    });
  }

  // ── canvas selection → inputs ─────────────────────────────────────────────────
  // Selecting Load Image / Load Video node(s) on the ComfyUI graph feeds their
  // file(s) to the agent as inputs — same as attaching them — in selection order.
  _ensureSelectionTracking() {
    const canvas = app.canvas;
    if (!canvas || canvas.__agentYSelHook) return;
    canvas.__agentYSelHook = true;
    const prev = canvas.onSelectionChange;
    canvas.onSelectionChange = (sel) => {
      try { if (prev) prev.call(canvas, sel); } catch (_) {}
      try {
        const ids = sel ? Object.keys(sel).map(Number) : [];
        const idset = new Set(ids);
        const prevSet = new Set(this._selOrder);
        // Drop deselected, then append newly-selected in the order they appear.
        // For click-by-click selection each change adds exactly one node, so the
        // resulting order is the true selection order.
        this._selOrder = this._selOrder.filter((id) => idset.has(id));
        for (const id of ids) {
          if (!this._selOrder.includes(id)) this._selOrder.push(id);
          if (!prevSet.has(id)) delete this._consumed[id]; // re-selecting re-arms a node
        }
      } catch (_) {}
    };
  }

  _orderedSelectedNodes() {
    this._ensureSelectionTracking();
    const canvas = app.canvas, graph = app.graph;
    if (!canvas || !graph) return [];
    const selIds = new Set();
    if (canvas.selected_nodes) for (const k of Object.keys(canvas.selected_nodes)) selIds.add(Number(k));
    if (canvas.selectedItems && canvas.selectedItems.forEach)
      canvas.selectedItems.forEach((it) => { if (it && it.id != null && it.widgets !== undefined) selIds.add(Number(it.id)); });
    if (selIds.size === 0 && graph._nodes) for (const n of graph._nodes) if (n && n.is_selected) selIds.add(Number(n.id));
    const ordered = [], seen = new Set();
    const getNode = (id) => (graph.getNodeById ? graph.getNodeById(id) : (graph._nodes || []).find((n) => Number(n.id) === id));
    for (const id of (this._selOrder || [])) if (selIds.has(id) && !seen.has(id)) { seen.add(id); const n = getNode(id); if (n) ordered.push(n); }
    for (const id of selIds) if (!seen.has(id)) { seen.add(id); const n = getNode(id); if (n) ordered.push(n); }
    return ordered;
  }

  _loaderInfo(node) {
    const t = String((node && (node.type || node.comfyClass)) || "");
    if (!/load/i.test(t)) return null;
    const widgets = node.widgets || [];
    const get = (names) => {
      for (const nm of names) {
        const w = widgets.find((x) => x && x.name === nm && x.value != null && String(x.value).trim() !== "");
        if (w) return String(w.value);
      }
      return null;
    };
    if (/video/i.test(t)) {
      const v = get(["video", "file", "path", "filename"]);
      if (v) return { value: v, kind: "video", name: node.title || t };
    }
    const iv = get(["image", "file", "filename"]);
    if (iv) return { value: iv, kind: "image", name: node.title || t };
    const vv = get(["video", "path"]);
    if (vv) return { value: vv, kind: "video", name: node.title || t };
    return null;
  }

  _collectCanvasInputs() {
    const out = [];
    for (const n of this._orderedSelectedNodes()) {
      const info = this._loaderInfo(n);
      if (!info) continue;
      // Skip a still-selected node whose file was already sent unchanged, so a
      // follow-up message doesn't silently re-attach it (attach-once semantics).
      if (this._consumed[n.id] === info.value) continue;
      info._nodeId = n.id;
      out.push(info);
    }
    return out;
  }

  // Snapshot every selected node (ANY type) with its widget parameter values, so
  // the agent can read and — via set_canvas_node_params → the canvas_patch SSE
  // event — write back arbitrary parameters (e.g. read/alter a prompt node).
  _collectCanvasSelection() {
    const out = [];
    for (const n of this._orderedSelectedNodes()) {
      const widgets = this._widgetSnapshot(n);
      if (!Object.keys(widgets).length) continue; // nothing readable/editable
      out.push({
        id: String(n.id),
        type: String((n && (n.type || n.comfyClass)) || ""),
        title: String((n && n.title) || ""),
        widgets,
      });
    }
    return out;
  }

  // Apply an agent-initiated node edit to the live graph (no refresh, no re-queue).
  _applyCanvasPatch(ev) {
    const graph = app.graph;
    const nid = Number(ev.node_id);
    const node = graph && (graph.getNodeById
      ? graph.getNodeById(nid)
      : (graph._nodes || []).find((n) => Number(n.id) === nid));
    if (!node) {
      this._sys(`⚠️ Could not apply edit — node #${ev.node_id} is no longer on the canvas.`);
      return;
    }
    const params = ev.params || {};
    const applied = [];
    for (const [name, value] of Object.entries(params)) {
      const w = (node.widgets || []).find((x) => x && x.name === name);
      if (!w) continue; // unknown widget on this node — skip
      // Keep combo widgets valid: register a new option value if needed.
      if (w.options && Array.isArray(w.options.values) &&
          typeof value !== "object" && !w.options.values.includes(value)) {
        w.options.values.push(value);
      }
      w.value = value;
      try { if (w.callback) w.callback(value, app.canvas, node); } catch (_) {}
      applied.push(name);
    }
    app.graph.setDirtyCanvas(true, true);
    const title = (node.title || node.type || ("#" + ev.node_id));
    if (applied.length) {
      this._sys(`✏️ Updated **${title}** — set ${applied.map((a) => "`" + a + "`").join(", ")}.`);
    } else {
      this._sys(`⚠️ No matching widget on **${title}** to update.`);
    }
  }

  // ── canvas hooks (AgentYHook nodes) ──────────────────────────────────────────
  _hookNodes() {
    const graph = app.graph;
    if (!graph || !graph._nodes) return [];
    return graph._nodes.filter(
      (n) => n && (n.type === "AgentYHook" || n.comfyClass === "AgentYHook")
    );
  }

  // Follow a hook's "anchor" input link back to the node it's attached to.
  _anchorFor(hookNode) {
    const graph = app.graph;
    if (!graph) return null;
    const inputs = hookNode.inputs || [];
    let idx = inputs.findIndex((i) => i && i.name === "anchor");
    if (idx < 0) idx = 0;
    const inp = inputs[idx];
    if (!inp || inp.link == null) return null;
    const link = graph.links ? graph.links[inp.link] : null;
    if (!link) return null;
    return graph.getNodeById ? graph.getNodeById(link.origin_id) : null;
  }

  // Scalar widget values of a node (numbers/strings), for the [CANVAS HOOKS] block.
  _widgetSnapshot(node) {
    const out = {};
    for (const w of node.widgets || []) {
      if (w && w.name != null && w.value != null && typeof w.value !== "object")
        out[w.name] = w.value;
    }
    return out;
  }

  _collectCanvasHooks() {
    const hooks = [];
    for (const hn of this._hookNodes()) {
      const w = this._widgetSnapshot(hn);
      if (w.ignore === true || w.ignore === "true") continue; // hook disabled — skip it
      const directive = String(w.directive || "").trim();
      if (!directive) continue; // an empty hook is a no-op
      const anchor = this._anchorFor(hn);
      // A hook wired FROM another hook is a downstream stage in a chain: its
      // input is the predecessor's output (resolved at run time), so record
      // prev_hook_id and leave anchor_node_id null. When the anchor is a real
      // node (the common case) behavior is unchanged.
      const anchorIsHook = !!anchor &&
        (anchor.type === "AgentYHook" || anchor.comfyClass === "AgentYHook");
      const realAnchor = anchor && !anchorIsHook ? anchor : null;
      hooks.push({
        hook_node_id: String(hn.id),
        directive,
        purpose: String(w.purpose || "directive"),
        mode: String(w.mode || "auto"),
        prev_hook_id: anchorIsHook ? String(anchor.id) : null,
        anchor_node_id: realAnchor ? String(realAnchor.id) : null,
        anchor_type: realAnchor ? String(realAnchor.type || realAnchor.comfyClass || "") : null,
        anchor_title: realAnchor ? String(realAnchor.title || "") : null,
        anchor_widgets: realAnchor ? this._widgetSnapshot(realAnchor) : {},
      });
    }
    return hooks;
  }

  // Capture the current graph as an API-format prompt (node-id keyed). Async in
  // recent ComfyUI (returns a promise); awaiting a plain object is also fine.
  async _captureCanvasGraph() {
    try {
      const p = await app.graphToPrompt();
      return p && p.output ? p.output : null;
    } catch (e) {
      return null;
    }
  }

  // ── slash-command popup ──────────────────────────────────────────────────────
  _onInput() {
    this._autosize();
    const v = this.input.value;
    if (v === "/") this._showPop("");
    else if (v.startsWith("/") && !v.includes(" ")) this._showPop(v.slice(1));
    else this._hidePop();
  }
  _autosize() {
    this.input.style.height = "auto";
    this.input.style.height = Math.min(this.input.scrollHeight, 140) + "px";
  }
  _showPop(q) {
    this._filtered = this.commands.filter((c) => c.name.slice(1).startsWith(q));
    if (!this._filtered.length) return this._hidePop();
    this._popSel = 0;
    this._renderPop();
    this.pop.style.display = "block";
  }
  _renderPop() {
    this.pop.innerHTML = "";
    this._filtered.forEach((c, i) => {
      const item = el("div", { className: "ay-pop-item" + (i === this._popSel ? " sel" : "") }, [
        el("span", { className: "ay-pop-name", textContent: c.name }),
        el("span", { className: "ay-pop-desc", textContent: c.description }),
      ]);
      item.addEventListener("mousedown", (e) => { e.preventDefault(); this._pickCmd(c); });
      this.pop.append(item);
    });
  }
  _hidePop() { this.pop.style.display = "none"; }
  _pickCmd(c) {
    const needsArg = ["/switch_model", "/add_workflow", "/remove_workflow"].includes(c.name);
    this.input.value = c.name + (needsArg ? " " : "");
    this._hidePop();
    this.input.focus();
    if (!needsArg) { /* leave for the user to press Enter */ }
  }
  _onKeydown(e) {
    const popOpen = this.pop.style.display === "block";
    if (popOpen) {
      if (e.key === "ArrowDown") { e.preventDefault(); this._popSel = (this._popSel + 1) % this._filtered.length; this._renderPop(); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); this._popSel = (this._popSel - 1 + this._filtered.length) % this._filtered.length; this._renderPop(); return; }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) { e.preventDefault(); this._pickCmd(this._filtered[this._popSel]); return; }
      if (e.key === "Escape") { this._hidePop(); return; }
    }
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.send(); }
  }
}

// ── register the sidebar tab ────────────────────────────────────────────────────
app.registerExtension({
  name: "agentY.chat.sidebar",
  async setup() {
    const register = () => {
      if (!app.extensionManager || !app.extensionManager.registerSidebarTab) return false;
      app.extensionManager.registerSidebarTab({
        id: "agentY-chat",
        icon: "pi pi-comments",
        title: "agentY",
        tooltip: "Chat with agentY — generate/edit media as graph nodes",
        type: "custom",
        render: (elm) => { new AgentChat(elm); },
      });
      console.log("[agentY] chat sidebar tab registered");
      return true;
    };
    if (!register()) {
      console.warn("[agentY] extensionManager.registerSidebarTab unavailable — update ComfyUI frontend to use the chat sidebar.");
    }
  },
});
