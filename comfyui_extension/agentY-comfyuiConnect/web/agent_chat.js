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
  { name: "/add_workflow", description: "Add a ComfyUI workflow" },
  { name: "/resend", description: "Resend the first user message" },
  { name: "/remove_workflow", description: "Remove a workflow by name" },
];

class AgentChat {
  constructor(root) {
    this.root = root;
    this.threadId = null;
    this.streaming = false;
    this.activeAsk = null; // request_id awaiting a reply
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
    this._loadThreads();
    this.newThread();
  }

  // ── styling ────────────────────────────────────────────────────────────────
  _injectStyles() {
    if (document.getElementById("agentY-chat-styles")) return;
    const css = `
    .ay-wrap{display:flex;flex-direction:column;height:100%;font-family:system-ui,-apple-system,sans-serif;font-size:13px;color:var(--fg-color,#ddd);background:var(--comfy-menu-bg,#1e1e2e);}
    .ay-bar{display:flex;gap:6px;align-items:center;padding:6px 8px;border-bottom:1px solid #33344a;flex-shrink:0;}
    .ay-bar select{flex:1;background:#26263a;color:#ddd;border:1px solid #3a3a5c;border-radius:6px;padding:4px;}
    .ay-btn{background:#2d2d50;color:#cfd2ff;border:1px solid #3a3a5c;border-radius:6px;padding:4px 8px;cursor:pointer;font-size:12px;}
    .ay-btn:hover{background:#3a3a63;}
    .ay-log{flex:1;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:8px;}
    .ay-msg{padding:8px 10px;border-radius:8px;max-width:100%;word-wrap:break-word;line-height:1.4;}
    .ay-user{background:#2b3a5b;align-self:flex-end;}
    .ay-assistant{background:#26263a;}
    .ay-system{background:#2a2320;color:#e8d9b0;font-size:12px;}
    .ay-ask{background:#3a2a10;color:#ffd27d;border:1px solid #6a4a10;}
    .ay-code{white-space:pre-wrap;font-family:monospace;background:#15151f;padding:2px 4px;border-radius:4px;}
    .ay-step{border:1px solid #33344a;border-radius:6px;background:#1b1b28;}
    .ay-step>summary{cursor:pointer;padding:6px 8px;color:#9da5ff;font-weight:600;}
    .ay-step .ay-step-body{padding:6px 10px;white-space:pre-wrap;font-family:monospace;font-size:11px;color:#9aa;max-height:220px;overflow:auto;}
    .ay-status{font-size:11px;color:#8a8;padding:2px 10px;font-family:monospace;}
    .ay-inwrap{border-top:1px solid #33344a;padding:8px;display:flex;flex-direction:column;gap:6px;flex-shrink:0;position:relative;}
    .ay-attach{display:flex;flex-wrap:wrap;gap:4px;}
    .ay-chip{background:#2d2d50;border:1px solid #3a3a5c;border-radius:4px;padding:2px 6px;font-size:11px;}
    .ay-inrow{display:flex;gap:6px;align-items:flex-end;}
    .ay-input{flex:1;resize:none;min-height:36px;max-height:140px;background:#26263a;color:#ddd;border:1px solid #3a3a5c;border-radius:8px;padding:8px;font-family:inherit;font-size:13px;}
    .ay-pop{position:absolute;bottom:100%;left:8px;right:8px;background:#1e1e2e;border:1px solid #3a3a5c;border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,.5);z-index:50;max-height:260px;overflow:auto;display:none;}
    .ay-pop-item{padding:7px 10px;cursor:pointer;display:flex;gap:10px;}
    .ay-pop-item.sel{background:#2d2d50;border-left:3px solid #7c83ff;}
    .ay-pop-name{font-family:monospace;color:#9da5ff;min-width:130px;}
    .ay-pop-desc{color:#888;font-size:12px;}
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

    this.sendBtn = el("button", { className: "ay-btn", textContent: "Send", style: { padding: "8px 14px" } });
    this.sendBtn.addEventListener("click", () => this.send());

    const inrow = el("div", { className: "ay-inrow" }, [attachBtn, this.input, this.sendBtn]);
    const inwrap = el("div", { className: "ay-inwrap" }, [this.pop, this.attachEl, inrow, this.fileInput]);
    wrap.append(inwrap);

    this.root.append(wrap);
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
    details.append(el("summary", { textContent: "▸ " + name }), body);
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
        // fold reasoning into a collapsible thinking step
        if (!this._thinkStep) { this._stepStart("💭 thinking"); this._thinkStep = this.curStep; }
        this._thinkStep.body.textContent += ev.data;
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
      case "system":
        this.curAssistant = null;
        this._sys(ev.data);
        break;
      case "ask":
        this.curAssistant = null;
        this.activeAsk = ev.request_id;
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
        this.streaming = false;
        this._setBusy(false);
        this._savePanel();  // persist the rendered panel so blocks survive reloads
        this._loadThreads();
        break;
    }
  }

  async _stream(body) {
    this.streaming = true;
    this._setBusy(true);
    try {
      const resp = await fetch(backendBase() + "/agentY/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
      this._sys("❌ Connection error: " + e + `\n\nIs the agentY chat host running? (\`run_agent.ps1\`, ${backendBase()})`);
    } finally {
      this.streaming = false;
      this._setBusy(false);
    }
  }

  _setBusy(b) {
    this.sendBtn.disabled = b && !this.activeAsk;
    this.sendBtn.textContent = b ? "…" : "Send";
  }

  // ── sending ──────────────────────────────────────────────────────────────────
  async send() {
    const text = this.input.value.trim();
    const canvasInputs = this._collectCanvasInputs();
    if (!text && this.attachments.length === 0 && canvasInputs.length === 0) return;

    // Answering an interactive ask → side-channel reply; the SSE stream continues.
    if (this.activeAsk) {
      const rid = this.activeAsk;
      this.activeAsk = null;
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
    this._userMsg(text + (noteParts.length ? `  \n_(${noteParts.join(", ")})_` : ""));
    this.input.value = "";
    this._autosize();
    this.attachments = [];
    this._renderAttachments();
    this._hidePop();
    // Mark these canvas files as consumed so an unchanged, still-selected node
    // isn't re-sent on the next message.
    for (const ci of canvasInputs) if (ci._nodeId != null) this._consumed[ci._nodeId] = ci.value;
    await this._stream({
      thread_id: this.threadId,
      message: text,
      image_paths: imgs,
      canvas_inputs: canvasInputs.map((c) => ({ value: c.value, kind: c.kind })),
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
