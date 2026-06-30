# workflow_recipes

Discover ComfyUI workflow *types* from a corpus of workflow JSON files and emit
a high-level **recipe database** that describes each type by node *roles and
relationships* - not literal copied subgraphs.

The database is a three-level hierarchy - **task -> model -> node clusters**:

```
Image to Video                         (task: the canonical category)
  +- WAN 2.2                           (model family)
  |    node clusters: model loading [UNETLoader x2, CLIPLoader, VAELoader],
  |    conditioning [CLIPTextEncode x2], sampling [KSamplerAdvanced x2], ...
  +- LTX-2
       node clusters: ...
```

Grouping by (task, model) makes each leaf recipe *tight* - same model means the
same loaders/samplers - so the node-cluster structure is precise rather than an
average over a whole task. Every leaf is **self-contained**: a populated
`description`, a `user_intent` block (media / task / model families /
when_to_use / example_requests), and the `node_clusters` (required structure).
A companion `node_knowledge.json` describes every node class (role + I/O
signature + which recipes use it).

It is meant to be consumed directly by a downstream LLM agent pipeline - a
"researcher" that maps a request like "build a video workflow using WAN 2.2" to
the task+model recipe, and a "brain" that wires the workflow to standard from
the node clusters - with no human annotation step.

This tool only *discovers* types and writes the database; it does not build
workflows, select recipes, wire nodes, or call any LLM.

## Quick start

```bash
# From the repo root (uses the two default template folders):
python -m workflow_recipes.cli --similarity-threshold 0.55

# Offline (use the cached object_info, never contact ComfyUI):
python -m workflow_recipes.cli --no-fetch

# Run the unit tests:
python -m unittest discover -s workflow_recipes/tests -p "test_*.py"
```

Outputs are written to `workflow_recipes/output/`:

| file | what it is |
|------|------------|
| `workflow_types.json` | the full recipe database, nested task -> model -> node clusters |
| `node_knowledge.json` | per node class: role, I/O signature, and which recipes use it |
| `workflow_types_report.md` | human-readable hierarchical report |
| `clustering_report.md` / `clustering_debug.json` | only written for the `description` / `structural` experimental modes |

## How it works (phases)

1. **parse** - load each file (UI or API format, auto-detected) into a
   normalized directed graph. ComfyUI subgraphs (a single node whose `type` is a
   UUID referencing `definitions.subgraphs`) are expanded *recursively*, with
   boundary ports rewired so connections that cross a subgraph boundary become
   direct edges. Node signatures are enriched from a cached `/object_info`
   response; unresolvable custom nodes are flagged, not fatal.
2. **fingerprint** - reduce each graph to structural signals that ignore
   parameter values and node-count noise: node-class set, typed connection
   patterns (`src_role -> dst_role [type]`), radius-1 local-cluster signatures,
   and which "spine" roles are present. Signals are weighted and individually
   droppable.
3. **group into types** (`--cluster-on`, default `category`):
   - **`category`** (default) - classify each workflow into the canonical
     ComfyUI task taxonomy (`taxonomy.py`) and group by it. One category = one
     type. Deterministic rule-based classification anchored on the catalog
     category, boundary-port modality, and authoritative API-node detection.
     This is what a user asks for ("Image to Video", "Image Edit with
     ControlNet", "API / Partner Nodes"), so it is the best researcher-matching
     basis. See the category list below.
   - **`description`** - TF-IDF cosine over the catalog descriptions + threshold
     agglomerative clustering. Groups by what reads alike (model family + task).
   - **`structural`** - weighted-Jaccard over node-graph fingerprints +
     threshold agglomerative. Groups by exact node-graph shape.

   The `description` and `structural` modes are threshold-based agglomerative
   (no preset cluster count; configurable threshold). All modes are deterministic.
4. **intent** - derive each workflow's `{media, task, model_families}` from
   filename tokens, catalog descriptions, node roles, and model-loader widget
   filenames, using transparent rule-based vocab tables (no LLM).
5. **recipe_builder** - synthesize one self-contained recipe per type. Highlights:
   - **Invariant detection**: a node class present in ALL members is a required
     structural invariant; present in only some members is optional/variant.
   - **Paired-node preservation**: a class consistently present 2+ times (e.g.
     the high-noise/low-noise UNETLoader pair in WAN 2.2) is surfaced with all
     instances required and never collapsed into one role.
   - **User intent + description**: a `user_intent` matching surface and an
     always-populated `description` (catalog text, else synthesized).
   - **Node knowledge**: `build_node_knowledge` emits per-class signatures.

## Key config flags

| flag | default | meaning |
|------|---------|---------|
| `--cluster-on` | `category` | grouping basis: `category` (canonical taxonomy), `description`, or `structural` |
| `--similarity-threshold` | `0.25` desc / `0.55` struct | merge clusters while avg similarity >= this (unused for `category`) |
| `--object-info-cache` | `workflow_recipes/object_info_cache.json` | read/written cache |
| `--templates-descriptions` | `config/workflow_templates.json` | flat name->description map enriching workflows the index.json files do not describe |
| `--host` / `--port` | `127.0.0.1` / `8188` | ComfyUI for `/object_info` |
| `--no-fetch` | off | never contact ComfyUI; cache only (offline) |
| `--weight-classes` | `0.40` | weight of the node-class signal |
| `--weight-connections` | `0.35` | weight of the connection-pattern signal |
| `--weight-clusters` | `0.20` | weight of the local-cluster signal |
| `--weight-spine` | `0.05` | weight of the spine-role signal |
| `--weight-category` | `0.0` | weight of the catalog-category signal (0 = off; see below) |
| `--custom-folder` / `--official-folder` | the two template folders | inputs |

### The canonical taxonomy (`--cluster-on category`, default)

`taxonomy.py` classifies every workflow into one of these categories (matching
ComfyUI's own template taxonomy at https://comfy.org/workflows):

> Text to Image | Image Edit | Image Edit with ControlNet | Inpaint / Outpaint |
> Upscale | Character | Image Tools | Text to Video | Image to Video |
> Video to Video | First / Last Frame to Video | Video Inpaint | Video Tools |
> 3D | Audio | Preprocessors / Estimation | Text Tools | API / Partner Nodes

Classification precedence: (1) text utilities (captioning / prompt), (2)
API/partner nodes (authoritative via the `comfy_api_nodes` module), (3) the
catalog category for coarse buckets (Image Tools, Video Tools, Audio, 3D,
Preprocessors, Image Editing), (4) refine the two broad generation buckets and
custom workflows by output media + input modality (from boundary ports) + name.
The category is also stored per type as `canonical_category`, and used for the
type `id`.

Partner/API workflows are functionally diverse, so they are sub-split by task -
`API / Partner Nodes - Image to Video`, `- Image Edit`, `- 3D`, `- Upscale`, ... -
keeping the recognizable partner prefix while each recipe stays coherent.

### The catalog-category signal (`--weight-category`, structural mode only)

By default clustering is purely structural (`--weight-category 0`). Raising this
weight lets the authoritative catalog category (from `index.json`) nudge the
grouping, so workflows in the same official category are pulled together. It is
**neutral** for any pair where either workflow has no catalog category (e.g.
custom workflows): the signal is dropped from that pair's weighted average
rather than scored as a match or a mismatch, so it never collapses uncategorized
graphs together. On this corpus, raising it from 0 -> 0.5 takes 68 -> ~56 types
at threshold 0.55. Tune it alongside `--similarity-threshold`.

## Recipe schema (`workflow_types.json`)

The top level is `tasks[]`; each task has `models[]`; each model is a recipe leaf.

```jsonc
{
  "structure": "task -> model -> node clusters",
  "tasks": [{
    "task": "Image to Video", "id": "image_to_video",
    "member_count": 7, "model_count": 2,
    "models": [{
      "id": "image_to_video__wan_2_2",   // {task}__{model} slug (deterministic)
      "model": "WAN 2.2",
      "user_intent": {                   // the researcher's matching surface
        "media": "video", "task": "image_to_video",
        "model_families": ["WAN 2.2"],
        "when_to_use": "Use to generate a video from an input image using WAN 2.2.",
        "example_requests": ["build a video workflow using WAN 2.2", "..."]
      },
      "description": "Image-to-video with Wan 2.2 ...",  // ALWAYS populated
      "description_source": "catalog",   // catalog | catalog+synthesized | synthesized
      "execution": "local",              // local | api (remote partner nodes) | hybrid
      "uses_api_nodes": false,           // any comfy_api_nodes node present
      "api_node_classes": [],            // which partner-node classes, if any
      "source": "custom | official | mixed",
      "member_files": ["..."],
      "member_descriptions": [{"name": "...", "title": "...", "description": "..."}],
      "member_count": 3,
      "node_clusters": [                 // required structure, grouped by function
        {"cluster": "model loading", "nodes": ["UNETLoader (x2)", "CLIPLoader", "VAELoader"]},
        {"cluster": "conditioning", "nodes": ["CLIPTextEncode (x2)"]},
        {"cluster": "sampling", "nodes": ["KSamplerAdvanced (x2)"]}
      ],
      // ... and the detailed fields below (roles / connections / ports):
      "required_node_roles": [              // present in ALL members (invariants)
    {
      "role": "diffusion model / UNET loader",
      "role_key": "model_loader",
      "node_class": "UNETLoader",
      "utility": false,                  // true => plumbing (primitive/math/switch)
      "frequency": "all members (3/3), 2 required instances",
      "min_instances": 2,                // guaranteed count across members
      "max_instances": 2,
      "paired_or_multiple": true,        // set only for meaningful 2+ instances
      "distinct_instances": [            // structural contexts that tell them apart
        {"feeds_into": ["sampler"], "fed_by": [], "occurrences": 3}
      ]
    }
  ],
      "optional_node_roles": [ ... ],    // present in SOME members (variant)
      "connection_patterns": [
        {"from_role": "model_loader", "to_role": "sampler",
         "data_type": "MODEL", "frequency": "all members (3/3)", "invariant": true}
      ],
      "boundary_ports": {
        "inputs":  [{"data_type": "IMAGE", "role": "image_loader"}],
        "outputs": [{"data_type": "VIDEO", "role": "save_output"}]
      },
      "param_variability": "varies across members: KSamplerAdvanced; constant: ...",
      "catalog_category": { "primary": "Video generation and editing", "...": "..." },
      "unresolved_nodes": [ ... ],       // classes absent from object_info
      "custom_nodes": [ ... ]            // resolved but third-party
    }]
  }]
}
```

Each leaf is self-contained; the earlier human-in-the-loop fields
(`needs_annotation` / `annotation_reason` / `notes_for_annotation`) were removed.
The `description` / `structural` modes instead emit a flat `types[]` list (same
leaf fields, grouped by similarity rather than task+model).

## `node_knowledge.json`

A companion database so the wiring brain knows each node's contract. One entry
per node class actually used in the corpus:

```jsonc
{
  "class": "KSampler",
  "role": "sampler",
  "role_description": "diffusion sampler / denoiser",
  "resolved": true, "is_custom": false,
  "inputs": {"required": ["model", "positive", "..."], "optional": [],
             "types": {"model": "MODEL", "positive": "CONDITIONING"}},
  "outputs": ["LATENT"],
  "used_in_type_ids": ["image_to_video_wan_2_2", "..."],
  "occurrences": 40
}
```

### Description sources (index.json + workflow_templates.json)

Authoritative descriptions come from two files, merged by `load_descriptions`:

- each folder's `index.json` - the official catalog gives a human **category**
  and **description** per workflow (custom index carries names only);
- `config/workflow_templates.json` - a flat `name -> description` map that fills
  the custom workflows the indexes do not describe.

Together these cover most of the corpus; the rest get a description synthesized
from intent + structure. This metadata is attached per member (`category`,
`suggested_title`, `member_descriptions`), aggregated per type (`category` with
`spans_multiple` flagging clusters that cross >=2 catalog categories), and used
for the type `description` (see below).

The catalog does **not** drive clustering - it enriches and validates the
structural groups. The official categories are deliberately coarse (e.g. "Image
generation and editing" spans 31 structurally distinct workflows), so using them
to group would erase the distinctions that make recipes useful.

### `description` policy (always populated, no annotation step)

1. Any member has an authoritative catalog description: it is used verbatim
   (`description_source: catalog`), even for custom-node types - these are
   human-authored. Differing member descriptions are joined.
2. Some members described, some not: catalog text is kept
   (`description_source: catalog+synthesized`).
3. No member described: a factual description is synthesized from the derived
   intent + structural spine (`description_source: synthesized`).

There is no human-in-the-loop flag - the database is finished as written.

### `user_intent` (the matching surface)

`intent.py` derives `{media, task, model_families}` per workflow from filename
tokens, catalog descriptions, node roles, and model-loader widget filenames, via
transparent vocab tables. At the type level these aggregate to a `when_to_use`
sentence and `example_requests` (e.g. "build a video workflow using WAN 2.2") so
the researcher can match a free-text request to a recipe. All rule-based and
tunable - edit the vocab tables in `intent.py` to adjust.

## Design notes

- **Stdlib only** (plus a `urllib` call for `object_info`). No `networkx` or ML
  clustering deps - the corpus is a few hundred small graphs, kept transparent
  and tunable.
- **Determinism**: same inputs + same threshold + same weights => identical
  clusters, ids, and output ordering across runs.
- **Roles vs raw classes**: connection patterns and roles use functional role
  labels so different-but-equivalent classes match; when a node has no known
  role its class name is kept so custom nodes stay distinguishable.
- Slugs (`id`) are derived from the intent (task + primary model family, e.g.
  `image_to_video_wan_2_2`), falling back to a structural signature when intent
  is uninformative. Deterministic and de-duplicated.
