# Generating with DashScope (Model Studio) directly

`dashscope_generate` / `dashscope_fetch` (`agenty_core/tools/dashscope_generate.py`, also in
agentY-mcp) call Alibaba's generation APIs with the user's `DASHSCOPE_API_KEY`, so the cost lands on
the DashScope bill and not on Comfy credits. Files go to ComfyUI's `agent/images` / `agent/videos`
with an `.agenty.json` sidecar.

## Endpoint
The native API needs a pay-as-you-go key (Token Plan / Coding Plan have none): `DASHSCOPE_PAYG_API_KEY`
with `DASHSCOPE_PAYG_BASE_URL`, else `DASHSCOPE_API_KEY` with `DASHSCOPE_BASE_URL`. The compatible-mode
URL is reduced to its host and `/api/v1` is appended (`agenty_core.dashscope_endpoints.native_url`);
without any URL the default is Singapore, `https://dashscope-intl.aliyuncs.com/api/v1`. Both the
workspace host (`ws-....maas.aliyuncs.com`) and the intl host were tested and behave the same. See
"DashScope: key, plan and endpoint must match" in `docs/reference.md`.

| Step | Request |
|---|---|
| create video task | `POST /services/aigc/video-generation/video-synthesis`, header `X-DashScope-Async: enable` |
| create async image task | `POST /services/aigc/text2image/image-synthesis`, same header |
| image (sync) | `POST /services/aigc/multimodal-generation/generation` (`input.messages`) |
| poll | `GET /tasks/{task_id}` (do not send the async header on a GET: a workspace host answers 403) -> `output.task_status` PENDING/RUNNING/SUCCEEDED/FAILED/CANCELED/UNKNOWN; result `output.video_url` / `output.results[].url`, valid 24 h; `usage` |

Input images: a public URL or a `data:image/png;base64,...` URL. Base64 was verified live (Wan 2.2 i2v).

## Models
"Accepted" = the international endpoint took the name (an unknown name returns `Model not exist`)
and the request shape passed validation. Shapes were checked by sending deliberately invalid
parameters (duration 999) and reading the validation message, which costs nothing. Only the models
marked "live OK" were actually run.

**Video** (`input` / `parameters`):

| Model | Task | Shape | Status |
|---|---|---|---|
| `wan2.7-t2v` `wan2.7-i2v` `wan2.7-r2v` | t2v / i2v / r2v | `media[{type: first_frame / reference_image, url}]`, `resolution` 720P/1080P, `ratio`, `duration` 2-15 | accepted |
| `wan2.6-t2v` `wan2.6-i2v` `wan2.6-i2v-flash` | t2v / i2v | i2v `input.img_url`; t2v `parameters.size` "W*H"; `duration` 2-15 | accepted |
| `wan2.6-r2v` `wan2.6-r2v-flash` | r2v | `input.reference_urls`, `size`, `duration` 2-10 | accepted |
| `wan2.5-t2v-preview` `wan2.5-i2v-preview` | | as wan2.6 | accepted |
| `wan2.2-i2v-flash` | i2v | `img_url`, `resolution` 480P/720P/1080P, fixed 5 s (no `duration`) | **live OK** |
| `wan2.2-t2v-plus` `wan2.1-t2v-turbo` `wan2.1-i2v-turbo` | | `size`; "duration customization is not supported" | accepted |
| `happyhorse-1.0-t2v` / `-i2v` / `-r2v` / `-video-edit` | | `media`, `resolution` 720P/1080P, `ratio` (not i2v), `duration` 3-15, no negative prompt | accepted |
| `happyhorse-1.1-t2v` | t2v | no documentation found | accepted |

HappyHorse **does exist on the international endpoint** (tested with this account). Not wired:
`-video-edit`, `wan2.2-kf2v-flash` (first + last frame), Wan 2.7 video continuation and audio-driven inputs.

**Image:**

| Model | Path | Status |
|---|---|---|
| `qwen-image-2.0`, `qwen-image-2.0-pro` | sync, `input.messages[{role: user, content: [{image}..., {text}]}]`, `parameters.size` "W*H", `n`, `watermark`; response `output.choices[0].message.content[].image`, `usage{width,height,image_count}`; images in `content` mean edit | `-2.0` **live OK** (512*512, 5 s) |
| `wan2.6-t2i`, `wan2.7-image`, `z-image-turbo` | sync, same shape | accepted |
| `wan2.5-t2i-preview`, `wan2.2-t2i-flash` | async text2image, `n` 1-4, result `output.results[].url` | accepted |
| `qwen-image-max`, `qwen-image-plus`, `qwen-image`, `qwen-image-edit*` | | **"Model not exist" on this account/endpoint** (editing is done by `qwen-image-2.0`) |

## Errors
`Model not exist` -> `unknown_model`; `Arrearage` / "good standing" / quota -> `balance`;
`DataInspectionFailed`, `IPInfringementSuspect` -> `content_refusal` (also known to
`src/utils/content_policy.py`); 401 / `InvalidApiKey` -> `auth`; 429 / `Throttling*` -> `rate_limit`;
unfinished after the timeout -> `timeout` with the `task_id` (`dashscope_fetch` collects it).
The balance and refusal patterns come from DashScope's documented error codes; they were not
triggered live.

## Live test (2026-10-19, intl host)
- `qwen-image-2.0`, 512*512: 5.3 s, usage `{image_count: 1, width: 512, height: 512}`.
- `wan2.2-i2v-flash`, 480P from a local PNG sent as base64: 34.5 s, usage `{duration: 5, video_count: 1, SR: 480}`.

## Cost: open
Usage is returned to the agent but agentY's cost overview only prices LLM tokens
(`compute_turn_cost` in `src/pipeline.py`); there is no hook for per-call non-LLM costs. Prices are
not hard-coded because the official price page is JS-rendered and could not be read. One
third-party mirror of the Alibaba Qwen-Image 2.0 Pro page shows 0.55 CNY/image on the Singapore
scope (unverified). TODO: a per-turn "external cost" list fed from `usage`, with prices from
https://www.alibabacloud.com/help/en/model-studio/model-pricing.

## Sources
- https://www.alibabacloud.com/help/en/model-studio/image-to-video-api-reference (Wan i2v)
- https://www.alibabacloud.com/help/en/model-studio/qwen-image-2-0-pro (Qwen-Image 2.0)
- https://help.aliyun.com/zh/model-studio/happyhorse-image-to-video-api-reference (and `-text-to-video`, `-reference-to-video`)
- Request shapes cross-checked against github.com/cinience/alicloud-skills (community mirror of those
  pages) and against the live API. The official pages could not be fetched directly.
