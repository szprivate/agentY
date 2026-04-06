/**
 * agentY bridge — ComfyUI frontend extension
 *
 * Features:
 *  - Polls /agentY/pending_previews every second and injects PreviewImage nodes
 *    for each pending job (or removes them when the server signals clear).
 *  - Registers the agentY/SendToAgent node type with an IMAGE input, a
 *    multiline message field, a read-only response field, and a
 *    "Send for Review" button that POSTs {graph_context, node_id, message} to
 *    /agentY/review.
 *  - Polls /agentY/node_responses every second and writes agent output into
 *    the matching node's response widget.
 *  - Server base URL is configurable via Settings → agentY → Server URL
 *    (key: agentY.serverUrl, default: http://localhost:5000).
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// ── Settings helpers ──────────────────────────────────────────────────────────

const SETTING_ID = "agentY.serverUrl";
const DEFAULT_URL = "http://localhost:5000";

function getServerUrl() {
    try {
        const val = app.extensionManager.setting.get(SETTING_ID);
        return (val && val.trim()) ? val.trim().replace(/\/$/, "") : DEFAULT_URL;
    } catch (_) {
        return DEFAULT_URL;
    }
}

// ── Injected-node registry (preview jobs) ────────────────────────────────────
// Maps job_id (string) → LiteGraph node instance

const _injected = new Map();

// ── SendToAgent node registry ─────────────────────────────────────────────────
// Maps node.id (number) → SendToAgentNode instance — maintained via onAdded/onRemoved

const _sendToAgentNodes = new Map();

// ── Polling: pending previews ─────────────────────────────────────────────────

async function pollPendingPreviews() {
    let jobs;
    try {
        const resp = await fetch(`${getServerUrl()}/agentY/pending_previews`, {
            signal: AbortSignal.timeout(900),
        });
        if (!resp.ok) return;
        jobs = await resp.json();
    } catch (_) {
        return;
    }

    if (!Array.isArray(jobs)) return;

    for (const job of jobs) {
        if (job.clear && job.job_id) {
            const node = _injected.get(job.job_id);
            if (node) {
                app.graph.remove(node);
                _injected.delete(job.job_id);
            }
            continue;
        }

        if (!job.job_id || _injected.has(job.job_id)) continue;

        const node = LiteGraph.createNode("PreviewImage");
        if (!node) {
            console.warn("[agentY] LiteGraph.createNode('PreviewImage') returned null");
            continue;
        }

        node.title = `agentY: ${job.label ?? job.job_id}`;
        node.properties = node.properties ?? {};
        node.properties.agentY_job_id = job.job_id;

        if (Array.isArray(job.origin_pos) && job.origin_pos.length >= 2) {
            node.pos = [job.origin_pos[0] + 50, job.origin_pos[1] + 150];
        }

        app.graph.add(node);
        _injected.set(job.job_id, node);
        console.info(`[agentY] Injected PreviewImage node for job "${job.job_id}"`);
    }
}

// ── Polling: agent responses ──────────────────────────────────────────────────

async function pollNodeResponses() {
    if (_sendToAgentNodes.size === 0) return;

    let responses;
    try {
        const resp = await fetch(`${getServerUrl()}/agentY/node_responses`, {
            signal: AbortSignal.timeout(900),
        });
        if (!resp.ok) return;
        responses = await resp.json();
    } catch (_) {
        return;
    }

    for (const [nodeId, text] of Object.entries(responses)) {
        // node IDs are numbers in LiteGraph but strings in JSON keys
        const node = _sendToAgentNodes.get(Number(nodeId));
        if (!node) continue;

        const w = node.widgets?.find(w => w.name === "response");
        if (w) {
            w.value = text;
            // Resize node to fit the new content
            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true, false);
        }
    }
}

// ── SendToAgent node type ─────────────────────────────────────────────────────

class SendToAgentNode extends LiteGraph.LGraphNode {
    constructor() {
        super();
        this.title = "Send to agentY";
        this.addInput("image", "IMAGE");

        // Multiline message input
        ComfyWidgets["STRING"](this, "message", ["STRING", { multiline: true }], app);

        // Read-only response output — not serialized, updated by polling
        const { widget: responseWidget } = ComfyWidgets["STRING"](
            this, "response", ["STRING", { multiline: true }], app
        );
        responseWidget.inputEl.readOnly = true;
        responseWidget.inputEl.setAttribute("placeholder", "Agent response will appear here…");
        responseWidget.inputEl.style.opacity = "0.65";
        responseWidget.inputEl.style.cursor = "default";
        responseWidget.serialize = false;  // exclude from workflow widgets_values

        // Send button
        this.addWidget("button", "Send for Review", null, () => {
            this._sendForReview();
        });

        this.size = [280, 220];
    }

    onAdded() {
        _sendToAgentNodes.set(this.id, this);
    }

    onRemoved() {
        _sendToAgentNodes.delete(this.id);
    }

    async _sendForReview() {
        const baseUrl = getServerUrl();
        const messageWidget = this.widgets?.find(w => w.name === "message");
        const payload = {
            graph_context: app.graph.serialize(),
            node_id: this.id,
            message: messageWidget?.value ?? "",
        };
        try {
            const resp = await fetch(`${baseUrl}/agentY/review`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (resp.ok) {
                console.info(`[agentY] Review request submitted (node ${this.id})`);
            } else {
                console.warn(`[agentY] Review request returned HTTP ${resp.status}`);
            }
        } catch (err) {
            console.error("[agentY] Failed to send for review:", err);
        }
    }

    onExecute() {}
}

SendToAgentNode.title = "Send to agentY";
SendToAgentNode.comfyClass = "agentY/SendToAgent";

// ── Extension registration ────────────────────────────────────────────────────

app.registerExtension({
    name: "agentY.bridge",

    async setup() {
        app.extensionManager.registerSetting({
            id: SETTING_ID,
            name: "Server URL",
            category: ["agentY", "agentY", "Server URL"],
            type: "text",
            defaultValue: DEFAULT_URL,
            tooltip: "Base URL of the agentY bridge server (e.g. http://localhost:5000)",
        });

        LiteGraph.registerNodeType("agentY/SendToAgent", SendToAgentNode);

        setInterval(pollPendingPreviews, 1000);
        setInterval(pollNodeResponses, 1000);

        console.info("[agentY] Bridge extension loaded. Polling every 1 s.");
    },
});
