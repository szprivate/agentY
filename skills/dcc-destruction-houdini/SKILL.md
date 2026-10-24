---
name: dcc-destruction-houdini
description: Destroy a 3D model in Houdini with a rigid-body (RBD) simulation — a building, a set or an object collapsing, breaking apart, being knocked over — plus dust and simple cloth, rendered as an OpenGL flipbook (or Karma for a single frame). Use for "make the house collapse", "destroy this in Houdini", "RBD destruction", "fracture and simulate", "add dust to the collapse", "render the destruction from another camera".
allowed-tools: dcc_build video_frames media_info analyze_image run_script write_text_file read_text_file show_in_panel
---

# Destruction in Houdini (RBD, dust, cloth)

Houdini has no live mode: everything runs headless through `dcc_build(app="houdini", mode="batch")`
(in the `dcc` tool pack: `load_tools("dcc")` if it is not in your list). You write ONE Python script
for hython that builds the whole network, cooks it and writes caches and frames; every change is an
edit to that script and a new `dcc_build` call. Never hand-patch a scene.

## The script

One file, parameters at the top, steps you can switch on and off, cached per step. Per-run settings
(which steps, which camera, sim numbers) go in the `params` argument of `dcc_build`, which the script
reads with

    params = globals().get("DCC_PARAMS") or json.loads(os.environ.get("DCC_PARAMS", "{}"))
    STEPS = params.get("steps", "source,fracture,sim,flags,render,encode").split(",")

(`DCC_PARAMS_FILE` has the same JSON for large params.) Do not use side `.json` files or inherited
environment variables for this: an old file silently changes a later call. A camera or look change
re-runs only `render` and `encode` from the caches. Cache every step to `.bgeo.sc` files.

- `timeout_minutes` (default 15 min, max 480) for long calls; `frames="1-96"` renders a sequence
  through the OpenGL ROP when the script does not render itself.
- Pass a scratch `save_to` for render-only reruns, so the delivered `.hip` is not overwritten.
- Output only comes back when the run ends: write your own log file with a line per step.
- **A node error does not fail the call.** A VEX compile error shows up only as text in `stdout_tail`
  / `error` of a call that still says `ok: true` or fails later: read `stdout_tail` every time, and
  print `node.errors()` for the nodes you cook. In VEX always assign `prim()`, `point()`,
  `findattribval()` to a TYPED local first (`float a = prim(1,"x",i);`): used inside an expression
  they are ambiguous or silently read only the x component.
- Render and encode in SEPARATE calls: hython keeps the cache files open while rendering, and an
  encode in the same process failed with "Too many open files" (a truncated mp4). After encoding,
  verify the frame count (`media_info`, or `ffprobe -count_frames`), never trust the log line.
- Test on a few frames before every full run and look at them (`video_frames` makes a contact sheet
  from a video or a folder of frames).

## 1 Source

- Bring the model in as USD. Read the stage's up axis and `metersPerUnit`, convert to Y up, metres.
- USD mesh prims arrive as `<object>/<object>_NNN`: use the parent name as the piece name.
- A mesh of lap siding is one mesh with many boards: split it into connected components, each a piece.
- Frame 1 must show the intact model. Check it first.

## 2 Fracture: one material per structural element

`rbdmaterialfracture::3.0`, wood mode, **Fracture per Piece on** (else the grain follows the wrong
axis). Grain lines run along each piece's longest axis, cuts across it; splinter density/length pull
points off the cut planes; randomness on grain and cut spacing breaks the regularity. Look at the
building and give each element type its own settings (numbers from a 12 m stilt house):

| element | fracture |
|---|---|
| siding / cladding boards (thin) | one cut across (about 0.75 m), a grain split (0.09 m), short splinters |
| deck / floor | planks 0.27 m wide along the long axis, a cut across every 1.9 m |
| wall sheathing | panels about 1.1 m, light splinters |
| roof | panels about 1.3 m (between rafters), no grain split |
| posts, mast, girders (thick) | NOT equal segments: 1–2 jagged cuts (2.3–3.2 m spacing), 2–3 long slivers, the longest splinters |
| braces | one mid snap |
| glass | scatter points 5, radial cracks 8, min width 0.11 |
| gutters, rails, clock, antennas | stay whole |
| a triangular gable wall | stays whole (wood fracture fills it into a box / spikes) |

Rules that held: keep `wood_graindetailsize` at 0.15 or more (the default 0.05 used 50+ GB or hung);
keep splinters short on thin panels (they poke out of both faces); drop pieces whose centre is
outside their source box (stray debris); give every piece a unique name (`<source>__<fragment>`) and
per-piece attributes the constraints need (class, source, upper body or substructure, section).
Mark the fracture's cut faces with a prim attribute `cutface` (the faces in the node's `*inside*`
groups). More pieces cost sim time roughly linearly: 1,500 pieces, 0.3 s/frame; 3,800 pieces, 0.75 s/frame.

## 3 Joints follow how the building is built

`rbdconstraintsfromrules` (surface, small search radius) builds the bonds; a prim wrangle on the
constraint network (input 2 = the pieces) sets each bond's type and strength by PAIR TYPE. One network
mixes types per bond; Glue: `strength`, `impulse_halflife`, `propagate_rate`; Hard: `constraint_name`
"Hard", `constraint_type` "position" (pin) or "all"; Soft: `stiffness`, `dampingratio`,
`enableangular`, `angularstiffness`, `enableplasticity`, `plasticthreshold`.

- Inside one board / beam (same source): the strongest glue (material), breaks only along the
  fracture pattern.
- Board → framing behind it ("nails"): weak (about 0.35x), so boards tear off whole or in halves.
- Board → board (laps): about 0.6x.
- Post ↔ girder, brace ends (bolted joints): strong, but they are the progressively weakened ones
  (below), so braces pull out and the frame racks.
- The upper body (cabin, roof, deck): strong glue inside each section, spring (Soft) bonds between
  neighbouring sections of a long body so it bends and stays bent (plasticity), a very soft bond at a
  seam where it should tear. Glass is left out of strong glue so it always shatters.
- Posts that held (6x glue between their own segments) kept the racking; without it they crumbled.

## 4 How it fails (no visible push)

- Weakening glue alone does not make a braced structure fall (it stands on friction).
- Directed collapse under its own weight: ramp the joints' and bracing's strength down with `@Frame`
  (enable constraint updates on the solver), starting at one end and moving across (start time keyed
  to x, 20–40 frames end to end). That reads as a collapse front. Add a slight sideways gravity on
  the Bullet solver (`gravityx`, about 2 m/s² = 6° with a strong structure; 1 m/s² was enough for a
  weak one) to choose the direction, and FADE it out after the landing (an expression on the gravity
  parm), otherwise the wreck keeps sliding sideways.
- A visible external force (wave, ram) is a moving collider: a PACKED primitive with an animated packed
  transform, in the solver's FOURTH input, `collision_overwritefromsop` on. Animated points of an
  unpacked box do nothing. A static collider overlapping the structure props it up.
- Friction about 0.9, bounce about 0.05 so it doesn't rock back.

## 5 Impact-driven break-up (two passes)

Pass 1 is the baseline sim: record when each source part first hits the ground (a piece below 0.9 m
losing >= 1.6 m/s in a frame, read from the solver's per-piece output) and write `impacts.json`;
also the frame the house body lands (about the 15th percentile of its parts' first hits). Pass 2 is
the same sim with a per-part glue drop from its impact frame (3-frame ramp) and for the whole body from
its landing frame (a few frames of random spread). The structure holds exactly as in pass 1 until
then, so the lean and the rack are unchanged. Factors by pair type: joints between parts about 1.5%,
glue inside a part by element (boards 12%, posts/beams 30%, panels and roofs 6%). So boards tear off,
panels split, beams hold longer.

## 6 Intact until it breaks

Fracture leaves tan cut faces and splinter chips on the surface. Hide them until a part breaks: a
`breaks` step fits the best rigid motion (Kabsch / SVD) of each original member's points (group by the
piece name without its fragment suffix) and records the first frame the residual RMS exceeds about
0.05–0.12 m (0.04 flagged soft-joined sections that merely bend); at render, delete the `cutface`
prims of a member until that frame. Frame 1 is then spotless; broken parts show their splinters.

## 7 Solver

RBD Bullet Solver SOP, ground plane at y = 0, 4 substeps. It cooks frame by frame in order.
Outputs: geometry, the constraint network (bonds disappear as they break), per-piece point cloud
(output 4: position and velocity).

## 8 Dust and cloth

- Dust: events = bonds that disappeared between frames and pieces that hit the ground. Write them as
  one point file per frame (`v`, `pscale0`, `a0`). POP network (`popobject` → `popsolver::2.0`,
  `popsource::2.0` → `popwind` → `popwrangle` into solver input 2; `emittype=0`, `impulserate=1`,
  `constantactivate=0`); `popwind.turb` is an integer; read back with `dopimport`. Fade/grow in the
  wrangle. Render as discs. More puffs near a seam that tears.
- **Pyro was tried and dropped:** Pyro (sparse, smoke only, 0.14 m voxels) looked low-res and
  flickered in the OpenGL flipbook. Karma renders it properly but took about 250 s per frame
  (CPU; "XPU" fell back to CPU), too slow for a sequence. Use the POP discs.
- Flags / cloth: Vellum, coarse (about 200 points, 2 substeps). Attach to the RBD part with a second
  `vellumconstraints` set to attach (`constrainttype=7`): cloth in input 1, cloth constraints in
  input 2, the target (the mast read from the sim cache each frame) in input 3 and the same target in
  the `vellumsolver` input 3. `maxdistcheck=1` with a small `maxdist` (about 0.16 m); without it every
  point attaches. The `pin` constraint type did nothing here. Cook frame by frame in order.

## 9 Render (OpenGL flipbook)

- OpenGL ROP: about 0.1–0.4 s per frame, the default. It draws per-point colour (`Cd` on POINTS: make
  points unique per face and promote the prim colours), discs for particles (`particle=3`,
  `transparency=1`, `pscale`, point `Alpha`). A point `Alpha` applies to the whole merged geometry:
  give everything else `Alpha=1`. `pointsize=0` hides stray points; `backfacecull=0` for cloth.
- Low cameras: a shadowless fill light along the camera's view direction (else undersides are
  black), a ground plane thousands of metres wide (no edge), smaller dust near the lens.
- Several cameras: the camera is a node. A moving camera is Hscript `fit($FF, ...)` expressions on
  translate/rotate. A follow camera: a key per frame; the look-at is the Gaussian-smoothed centroid of
  the house body pieces from the sim cache (sigma about 9 frames) plus a small dolly. Re-render only the
  render step per camera from the cache, with its own output name.
- Karma for a still: about 80 s per frame at 960x540 CPU without volumes. Render ONE frame first and
  show it before a sequence.
- Encode with ffmpeg to mp4 (separate call), keep a few stills, delete the raw frames.

## 10 What to expect (about 3,800 pieces, 144 frames at 960x540)

Fracture 12 s; each of the two sim passes about 105 s; breaks step 20 s; Vellum flags 45 s; dust 3 s;
OpenGL render 35–50 s per camera; encode 1 s. A full re-run through `dcc_build` is about 3–4 minutes.

## 11 Show the result

Show a contact sheet (start, failure, lean, landing, settle) and say what reads and what does not.
Check frame 1 in a close-up, the break close-ups, the ends of the clip. Ask before a long final render.
