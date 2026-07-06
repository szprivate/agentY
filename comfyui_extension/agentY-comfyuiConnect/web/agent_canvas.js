import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// agentY-comfyuiConnect: when the agentY pipeline runs a workflow it POSTs the graph to
// /agent/load_workflow, which the server broadcasts as an "agent.load_workflow"
// websocket event. Load that graph onto the canvas so the user sees exactly what
// just ran, without clicking through the Workflows sidebar.
app.registerExtension({
  name: "agentY.canvas.autoload",
  async setup() {
    api.addEventListener("agent.load_workflow", async (event) => {
      const graph = event && event.detail;
      if (!graph || !Array.isArray(graph.nodes)) return;
      try {
        // clean=true, restore_view=true — replace the current graph with the run.
        await app.loadGraphData(graph, true, true, "agent workflow");
      } catch (err) {
        try {
          await app.loadGraphData(graph); // older signatures
        } catch (err2) {
          console.error("[agentY-comfyuiConnect] loadGraphData failed:", err2);
        }
      }
    });
    console.log("[agentY-comfyuiConnect] ready — workflows the agent runs will open here");
  },
});
