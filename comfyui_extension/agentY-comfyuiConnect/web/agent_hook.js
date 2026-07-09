import { app } from "../../scripts/app.js";

// Give the agentY hook node a distinct warm color so it's obvious it's an agent
// annotation and not part of the executing pipeline.
app.registerExtension({
  name: "agentY.hookNode",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "AgentYHook") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      this.color = "#5c3a28";
      this.bgcolor = "#3a2a20";
      if (!this.title || this.title === "AgentYHook") this.title = "agentY hook";
      this.size = [280, 210];
      return r;
    };
  },
});
