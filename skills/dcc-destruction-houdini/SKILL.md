---
name: dcc-destruction-houdini
description: Destroy a 3D model in Houdini with a rigid-body (RBD) simulation — a building, a set or an object collapsing, breaking apart, being knocked over — plus dust or smoke and simple cloth, rendered as a flipbook or with Karma. Use for "make the house collapse", "destroy this in Houdini", "RBD destruction", "fracture and simulate", "add dust / smoke to the collapse", "render the destruction from another camera".
allowed-tools: dcc_build video_frames media_info analyze_image run_script write_text_file read_text_file show_in_panel
---

# Destruction in Houdini (RBD, dust, smoke, cloth)

Houdini has no live mode: everything runs headless through `dcc_build(app="houdini", mode="batch")`
(in the `dcc` tool pack: `load_tools("dcc")` if it is not in your list). You write ONE Python script
for hython that builds the whole network, cooks it and writes caches and frames; every change is an
edit to that script and a new `dcc_build` call. Never hand-patch a scene.

## The script

One file, parameters at the top, steps you can switch on and off, e.g. through an environment
variable the script reads (`dcc_build` passes its environment to hython):

    STEPS = os.environ.get("SHOT_STEPS", "source,fracture,sim,dust,cloth,render,encode").split(",")

so a camera or look change re-runs only `render,encode` from the caches. Cache every step to
`.bgeo.sc` files in an output folder; never keep a sim only in memory.

- Pass `timeout_minutes` to `dcc_build` (fracture + sim + render easily exceed the default).
- Pass a scratch `save_to` for render-only reruns, so the delivered `.hip` is not overwritten.
- Houdini's output only comes back when the run ends: write your own log file with a line per step
  and per N frames (time, piece count, bbox), and read it when a run takes long or hangs.
- Test on 3 frames (start, middle, end) before every full run. Look at them yourself
  (`video_frames` / `analyze_image`) before you render the sequence.

## 1 Source

- Bring the model in as USD (export from Blender / Maya with `dcc_build`). Read the stage's up axis
  and `metersPerUnit` and convert to Houdini's Y up, metres.
- USD mesh prims arrive as `<object>/<object>_NNN`: use the parent name as the piece name.
- Frame 1 must show the intact model. Check it before anything else.

## 2 Fracture (before the sim)

- `rbdmaterialfracture` per material class: wood for walls / decks / beams (grain along each member,
  cuts across it), glass for windows (many small shards), concrete for masonry.
- Keep wood fracture coarse: `wood_graindetailsize` around 0.3. The default 0.05 can use 50+ GB of
  RAM or hang. Cap splinter length and noise, or the rest pose already looks shattered.
- Do not fracture long structural members (posts, beams, masts) into segments when they should act
  as members: keep them whole and let the joints break.
- Small parts (gutters, clocks, antennas, rails) stay whole. A wall that fractures into spikes stays
  whole or gets a coarse cut (few big pieces, no splinters).
- Give every piece a unique name traceable to its source mesh, and tag it (class, which part of the
  building, upper body vs. substructure) for the constraints.

## 3 Constraints: what holds, what gives

`rbdconstraintsfromrules` builds bonds between touching pieces; a wrangle on the constraint network
sets each bond's type and strength. One network can mix types per bond:

- Glue: `constraint_name="Glue"`, `strength`, `impulse_halflife`, `propagate_rate`.
- Hard: `constraint_name="Hard"`, `constraint_type="position"` (pin joint, rotation free) or `"all"`.
- Soft: `constraint_name="Soft"`, `stiffness`, `dampingratio`, `enableangular`, `angularstiffness`,
  `enableplasticity`, `plasticthreshold`, `plasticrate` — use these between large sections of a
  body so it bends and keeps the bend (a house sagging over an edge).

Patterns that worked:
- Strong glue inside the part that should survive (e.g. each room of a house, x30–x40), much
  weaker in the substructure, and near zero along a seam where it should tear.
- Glass excluded from strong glue so it always shatters.

## 4 How it fails (direction without a visible push)

- Weakening glue alone does NOT make a braced structure fall: it stands on friction.
- To make it collapse under its own weight: ramp the strength of the joints and bracing down over
  time (`@Frame` in the constraint wrangle; enable constraint updates on the solver), starting at
  one end and moving across (start time keyed to x) — that reads as a collapse front.
- Pick the direction with a slight sideways gravity on the Bullet solver (`gravityx` about 1 m/s²):
  invisible as a push, but it drifts loose pieces a little (keep it small).
- A visible external force (a wave, a ram) is a moving collider: it must be a PACKED primitive with
  an animated packed transform (pack, then a transform SOP with expressions), plugged into the
  Bullet solver SOP's FOURTH input, with `collision_overwritefromsop` on. Animated points of an
  unpacked box do nothing; the third input ignores it. A static collider that overlaps the
  structure props it up: never leave one in the geometry.
- Lower friction / bounce settle: friction about 0.9, bounce about 0.05 so it doesn't rock back.

## 5 Solver

RBD Bullet Solver SOP, ground plane on at y = 0, a few substeps. It cooks frame by frame in order
(`hou.setFrame` for each frame). Useful outputs: the per-piece output (positions and velocities)
and the constraint-network output, whose bonds disappear as they break.

Typical times at about 2,000–3,000 pieces: fracture 10–20 s, sim 0.3 s per frame.

## 6 Dust and smoke

- Events: bonds that disappear between frames (break points) and pieces near the ground that lose
  a lot of speed in one frame (impacts). Write them per frame as points with `v`, size, opacity.
- Dust particles: a dopnet with `popobject` → `popsolver::2.0` (input 1), and `popsource::2.0` →
  `popwind` → `popwrangle` into solver input 2. `popsource` with `emittype=0` (all points),
  `impulserate=1`, `constantactivate=0` inherits the event points' attributes. `popwind.turb` is an
  integer. Read the result back with `dopimport`. Sand colour, rise then settle, fade with age.
- Smoke: Pyro (smoke only, no flame) sourced from the same events, coarse voxels (0.08–0.15 m),
  ground collision, light buoyancy and wind. Check how it renders on one frame first.

## 7 Cloth (flags, sails, tarps)

Vellum cloth, coarse (a few hundred points, 2 substeps). Attach it to an RBD-driven part with a
second `vellumconstraints` node set to attach (`constrainttype=7`): cloth in input 1, the cloth
constraints in input 2, the target geometry (read from the sim cache each frame) in input 3; feed
the same target to the `vellumsolver` input 3. Use `maxdistcheck=1` with a small `maxdist` so only
the points near the pole attach (without it every point attaches). Collide with the debris.

## 8 Render

- OpenGL ROP flipbook: about 0.1 s per frame — the default for iterating. It draws per-point colour
  (`Cd` on points; move prim colours onto unique points), discs for particles (`particle=3`,
  `transparency=1`, `pscale` as size, point `Alpha`). Point `Alpha` applies to the whole merged
  geometry: give everything else `Alpha=1` or it vanishes. `pointsize=0` hides stray points.
  `backfacecull=0` for cloth.
- Low cameras need a shadowless fill light (otherwise undersides are black), a very large ground
  plane (no visible edge) and smaller dust near the lens.
- Karma: a minute or more per frame on CPU at 960x540 for a scene like this. Render ONE frame first
  and show it before committing to a sequence.
- Several cameras: the camera is just a node; re-run only the render step per camera from the cache.
- Encode with ffmpeg to mp4, keep a few stills, delete the raw frames.

## 9 Show the result

Show a contact sheet of the clip (start, failure, mid-collapse, landing, settle) and say what reads
and what doesn't. Ask before a long final render.
