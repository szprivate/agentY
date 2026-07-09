import { app } from "../../scripts/app.js";

// agentY Application Settings — adds an entry to the ComfyUI Settings panel that
// opens a modal for editing the agent's auth keys (.env) and everything in
// config/settings.json. Reads/writes go through the agentY chat host
// (src/utils/agentY_server.py: GET/POST /agentY/settings). The settings.json
// save is comment-preserving (only changed leaves are rewritten).

const DEFAULT_PORT = 5000;
function backendBase() {
  return (
    localStorage.getItem("agentY_backend") ||
    `http://${location.hostname || "127.0.0.1"}:${DEFAULT_PORT}`
  );
}

function el(tag, props = {}, children = []) {
  const n = document.createElement(tag);
  Object.assign(n, props);
  if (props.style) Object.assign(n.style, props.style);
  for (const c of [].concat(children)) if (c != null) n.append(c);
  return n;
}

function isSecret(key) {
  return /KEY|TOKEN|SECRET|PASSWORD/i.test(key);
}

function injectStyles() {
  if (document.getElementById("agentY-settings-styles")) return;
  const css = `
  .ays-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;
    align-items:center;justify-content:center;font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;}
  .ays-card{background:#262624;color:#f2f0ea;border:1px solid rgba(240,235,225,.12);border-radius:14px;
    width:min(720px,92vw);max-height:88vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.6);}
  .ays-head{display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid rgba(240,235,225,.10);}
  .ays-head h2{font-size:15px;margin:0;font-weight:650;flex:1;}
  .ays-body{padding:14px 18px;overflow:auto;}
  .ays-foot{display:flex;gap:10px;justify-content:flex-end;padding:12px 18px;border-top:1px solid rgba(240,235,225,.10);}
  .ays-sec{margin-bottom:18px;}
  .ays-sec>h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#a8a39a;margin:0 0 8px;}
  .ays-row{display:flex;align-items:center;gap:10px;padding:4px 0;}
  .ays-label{flex:0 0 42%;font-size:12.5px;color:#d7d2c8;word-break:break-word;font-family:ui-monospace,monospace;}
  .ays-input{flex:1;min-width:0;background:#302f2c;color:#f2f0ea;border:1px solid rgba(240,235,225,.14);
    border-radius:8px;padding:6px 9px;font-size:12.5px;outline:none;}
  .ays-input:focus{border-color:rgba(217,119,87,.6);}
  input.ays-input[type=checkbox]{flex:0 0 auto;width:16px;height:16px;accent-color:#d97757;}
  .ays-group{border:1px solid rgba(240,235,225,.10);border-radius:10px;margin:8px 0;overflow:hidden;}
  .ays-grouphead{background:#302f2c;padding:7px 12px;font-size:12px;font-weight:600;color:#e9c9b6;
    font-family:ui-monospace,monospace;cursor:pointer;}
  .ays-groupbody{padding:6px 12px;}
  .ays-btn{background:#3b3936;color:#f2f0ea;border:1px solid rgba(240,235,225,.14);border-radius:9px;
    padding:8px 16px;cursor:pointer;font-size:12.5px;}
  .ays-btn:hover{background:#464440;}
  .ays-btn.primary{background:#d97757;color:#2a1810;border-color:transparent;font-weight:650;}
  .ays-btn.primary:hover{background:#c56647;}
  .ays-note{font-size:11.5px;color:#a8a39a;margin:2px 0 10px;}
  .ays-msg{font-size:12px;margin-right:auto;align-self:center;}
  .ays-toggle{display:flex;align-items:center;gap:6px;font-size:11.5px;color:#a8a39a;margin-bottom:8px;cursor:pointer;}
  `;
  document.head.append(el("style", { id: "agentY-settings-styles", textContent: css }));
}

function buildModelSelect(groups, current) {
  const sel = el("select", { className: "ays-input" });
  let matched = false;
  for (const [group, models] of Object.entries(groups || {})) {
    if (!models || !models.length) continue;
    const og = el("optgroup", { label: group });
    for (const [spec, label] of models) {
      const o = el("option", { value: spec, textContent: label });
      if (spec === current) { o.selected = true; matched = true; }
      og.append(o);
    }
    sel.append(og);
  }
  // Keep the current value selectable even if no vendor advertises it.
  if (!matched) {
    const o = el("option", { value: current, textContent: current + "  (current)" });
    o.selected = true;
    sel.insertBefore(o, sel.firstChild);
  }
  return sel;
}

// Recursively render a settings object into form controls, collecting leaf refs.
function buildSettingsForm(container, obj, modelGroups, pathPrefix, refs) {
  for (const [key, val] of Object.entries(obj)) {
    const path = pathPrefix.concat(key);
    if (val && typeof val === "object" && !Array.isArray(val)) {
      const body = el("div", { className: "ays-groupbody" });
      const head = el("div", { className: "ays-grouphead", textContent: "▾ " + key });
      head.addEventListener("click", () => {
        const hidden = body.style.display === "none";
        body.style.display = hidden ? "" : "none";
        head.textContent = (hidden ? "▾ " : "▸ ") + key;
      });
      container.append(el("div", { className: "ays-group" }, [head, body]));
      buildSettingsForm(body, val, modelGroups, path, refs);
      continue;
    }
    const row = el("div", { className: "ays-row" });
    row.append(el("label", { className: "ays-label", textContent: key }));
    let input;
    const underPipeline = path[0] === "llm" && path[1] === "pipeline";
    if (typeof val === "boolean") {
      input = el("input", { type: "checkbox", className: "ays-input" });
      input.checked = val;
      refs.push({ path, get: () => input.checked });
    } else if (typeof val === "number") {
      input = el("input", { type: "number", className: "ays-input", value: String(val) });
      refs.push({ path, get: () => { const n = Number(input.value); return Number.isNaN(n) ? val : n; } });
    } else if (Array.isArray(val)) {
      input = el("input", { type: "text", className: "ays-input", value: JSON.stringify(val) });
      refs.push({ path, get: () => { try { return JSON.parse(input.value); } catch (_) { return val; } } });
    } else if (underPipeline && modelGroups && Object.keys(modelGroups).length) {
      input = buildModelSelect(modelGroups, val == null ? "" : String(val));
      refs.push({ path, get: () => input.value });
    } else {
      input = el("input", { type: "text", className: "ays-input", value: val == null ? "" : String(val) });
      refs.push({ path, get: () => input.value });
    }
    row.append(input);
    container.append(row);
  }
}

// Reassemble the nested settings object from the collected leaf refs.
function collectSettings(refs) {
  const out = {};
  for (const { path, get } of refs) {
    let node = out;
    for (let i = 0; i < path.length - 1; i++) {
      node[path[i]] = node[path[i]] || {};
      node = node[path[i]];
    }
    node[path[path.length - 1]] = get();
  }
  return out;
}

async function openAgentYSettingsModal() {
  injectStyles();
  let data;
  try {
    const r = await fetch(backendBase() + "/agentY/settings");
    if (!r.ok) throw new Error("HTTP " + r.status);
    data = await r.json();
  } catch (e) {
    alert("Could not load agentY settings — is the chat host running?\n\n" + e);
    return;
  }

  const overlay = el("div", { className: "ays-overlay" });
  const body = el("div", { className: "ays-body" });

  // ── .env auth section ──
  const envInputs = {};
  const secretEls = [];
  const envSec = el("div", { className: "ays-sec" });
  envSec.append(el("h3", { textContent: "Authentication (.env)" }));
  envSec.append(el("div", { className: "ays-note", textContent: "API keys and host settings. Stored in .env on the agent host." }));
  const showToggle = el("input", { type: "checkbox" });
  const toggleLabel = el("label", { className: "ays-toggle" }, [showToggle, el("span", { textContent: "Show secret values" })]);
  showToggle.addEventListener("change", () => {
    for (const inp of secretEls) inp.type = showToggle.checked ? "text" : "password";
  });
  envSec.append(toggleLabel);
  for (const key of data.env_keys || Object.keys(data.env || {})) {
    const cur = (data.env || {})[key] || "";
    const secret = isSecret(key);
    const inp = el("input", { className: "ays-input", type: secret ? "password" : "text", value: cur });
    if (secret) secretEls.push(inp);
    envInputs[key] = { input: inp, original: cur };
    envSec.append(el("div", { className: "ays-row" }, [el("label", { className: "ays-label", textContent: key }), inp]));
  }
  body.append(envSec);

  // ── settings.json section ──
  const refs = [];
  const setSec = el("div", { className: "ays-sec" });
  setSec.append(el("h3", { textContent: "Application settings (config/settings.json)" }));
  setSec.append(el("div", { className: "ays-note", textContent: "Model per stage (llm ▸ pipeline), directories, and behaviour toggles. Comments are preserved on save; only changed values are written." }));
  const setForm = el("div");
  buildSettingsForm(setForm, data.settings || {}, data.model_groups || {}, [], refs);
  setSec.append(setForm);
  body.append(setSec);

  // ── footer ──
  const msg = el("div", { className: "ays-msg" });
  const cancelBtn = el("button", { className: "ays-btn", textContent: "Close" });
  const saveBtn = el("button", { className: "ays-btn primary", textContent: "Save" });
  const close = () => overlay.remove();
  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    msg.textContent = "Saving…";
    const envChanges = {};
    for (const [key, { input, original }] of Object.entries(envInputs)) {
      if (input.value !== original) envChanges[key] = input.value;
    }
    const payload = { env: envChanges, settings: collectSettings(refs) };
    try {
      const r = await fetch(backendBase() + "/agentY/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || "save failed");
      const parts = [];
      if (j.env_updated && j.env_updated.length) parts.push(`${j.env_updated.length} auth key(s)`);
      if (j.settings_updated && j.settings_updated.length) parts.push(`${j.settings_updated.length} setting(s)`);
      msg.textContent = parts.length ? "✅ Saved " + parts.join(", ") + ". Model changes apply on next /switch_model or restart." : "No changes to save.";
      // Refresh originals so a second save doesn't re-send unchanged keys.
      for (const [key, o] of Object.entries(envInputs)) o.original = o.input.value;
    } catch (e) {
      msg.textContent = "❌ " + e;
    } finally {
      saveBtn.disabled = false;
    }
  });

  const foot = el("div", { className: "ays-foot" }, [msg, cancelBtn, saveBtn]);
  const card = el("div", { className: "ays-card" }, [
    el("div", { className: "ays-head" }, [el("h2", { textContent: "agentY — Application Settings" })]),
    body,
    foot,
  ]);
  overlay.append(card);
  document.body.append(overlay);
}

app.registerExtension({
  name: "agentY.settings",
  settings: [
    {
      id: "agentY.appSettings",
      name: "Application settings (auth keys + config)",
      category: ["agentY", "Application", "Settings"],
      tooltip: "Edit agentY auth keys (.env) and config/settings.json",
      defaultValue: "",
      // Custom render: a button that opens the agentY settings modal.
      type: (_name, _setter, _value) => {
        const btn = el("button", { className: "ays-btn", textContent: "Open agentY Settings…" });
        injectStyles();
        btn.addEventListener("click", (e) => { e.preventDefault(); openAgentYSettingsModal(); });
        return btn;
      },
    },
  ],
});
