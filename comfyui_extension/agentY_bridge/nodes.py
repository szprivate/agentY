"""
Python backend for the agentY/SendToAgent ComfyUI node.

When the graph is queued and executed, `passthrough` saves the image(s) to
temp PNG files and POSTs a review request to the agentY bridge server
(http://localhost:5000/agentY/review).  The message typed in the node's text
field is included in the payload so the agent can read it directly.

The "Send for Review" JS button does the same POST client-side, so either
path works — Run (queue) or button click.
"""

import os
import tempfile

import requests


_BRIDGE_URL = os.environ.get("AGENTY_BRIDGE_URL", "http://127.0.0.1:5000")


class SendToAgentNode:
    CATEGORY = "agentY"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "message": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "passthrough"
    OUTPUT_NODE = True

    def passthrough(self, image, message, unique_id=None):
        image_paths = self._save_images(image)
        self._post_review(unique_id, message, image_paths)
        return (image,)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _save_images(image) -> list[str]:
        """Save each image in the batch to a temp PNG and return the paths."""
        from PIL import Image as PILImage

        paths = []
        for i, img_tensor in enumerate(image):
            # img_tensor: [H, W, C] float32 0-1
            arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil_img = PILImage.fromarray(arr)
            tmp = tempfile.NamedTemporaryFile(
                prefix="agentY_review_",
                suffix=f"_{i}.png",
                delete=False,
            )
            pil_img.save(tmp.name, format="PNG")
            tmp.close()
            paths.append(tmp.name)
            print(f"[agentY] Saved review image: {tmp.name}")
        return paths

    @staticmethod
    def _post_review(node_id, message: str, image_paths: list[str]) -> None:
        payload = {
            "node_id": node_id,
            "message": message,
            "image_paths": image_paths,
            "source": "comfyui_execution",
        }
        try:
            resp = requests.post(
                f"{_BRIDGE_URL}/agentY/review",
                json=payload,
                timeout=5,
            )
            if resp.ok:
                print(f"[agentY] Review request queued (node {node_id})")
            else:
                print(f"[agentY] Bridge server returned HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(
                "[agentY] Could not reach bridge server at "
                f"{_BRIDGE_URL} — is agentY running?"
            )
        except Exception as exc:
            print(f"[agentY] Failed to send review request: {exc}")


NODE_CLASS_MAPPINGS = {
    "agentY/SendToAgent": SendToAgentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "agentY/SendToAgent": "Send to agentY",
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "agentY/SendToAgent": "Send to agentY",
}
