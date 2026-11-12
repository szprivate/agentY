---
name: character-animation
description: Rig and animate a character in the ComfyUI Image Viewer's 3D space - controls on a skinned model, auto-rig a model with no skeleton (UniRig), AI motion from a text prompt and AI in-betweens between keyed poses (NVIDIA Kimodo), feet pinned to the floor, motion restyled ("make the walk sneaky"), motion from a film, lip sync from a sound, AI takes on animation layers, constraints, the pose library, FBX / Alembic export. Use for "make the walk look tired", "restyle this as a march", "copy the dance from this video onto the hero", "lip sync her to this line", "put that on a new layer", "constrain the cup to his right hand", "save this pose", "apply the run pose", "pose him sitting with arms crossed", "make her kneel", "pose him like in this photo", "put her hand on the floor", "rig this character", "add controls", "auto-rig the creature", "make him walk forward and wave", "animate the hero from frame 0 to 96", "fill in the motion between my two poses", "in-between these keys", "clean up the foot sliding", "export the character as FBX / Alembic", "turn the takes into clips", "render a turntable of each clip", "add a camera looking at him", "key the cube at frame 24", "playblast the shot", "export the scene as USD".
allowed-tools: animate_character animate_scene render_turntables get_viewer_state show_in_viewer
---

# Characters in the 3D space

Everything happens in the viewer's open 3D scene (the browser must be open on it). `get_viewer_state()`
lists the scene's `characters`: name, whether each has a skeleton, a rig (humanoid or FK), and keyed frames.
Pick the model by its name; with one model in the scene the name can be left out.

1. **Rig.** A skinned model: `animate_character(action="rig", model="Hero")` puts controls on it - FK per
   bone, IK arms and legs with poles, a COG, when it reads as a humanoid; FK only otherwise (creatures,
   quadrupeds, props). A model with NO skeleton (a mesh from a 3D generator): `action="autorig"` runs
   UniRig on it (a skeleton and skin weights for any character, a minute or two) and rigs the result.
2. **Motion from text** (humanoids): `animate_character(action="motion", model="Hero", prompt="A person
   walks forward, stops and waves with the right hand.", from_frame=0, to_frame=120)`. Plain sentences,
   one per segment, a person as the subject. It starts where (and as) the character stands at
   `from_frame` and is keyed on the controls, so the user can edit it key by key.
3. **In-betweens** (humanoids): the user keys poses (Set Key in the 3D space); `action="inbetween"` with
   `from_frame` / `to_frame` around them fills the frames between the first and last key so every key in
   the range is hit. The two outer keys stay as they were. A prompt is optional ("a person jumps").
4. **Feet**: motion and in-betweens pin the standing feet and keep them out of the floor already;
   `action="clean_feet"` does it again over a range after hand edits.
5. **Export**: `action="export", format="fbx"` (skeleton, skin, keys on every frame - Maya, Houdini,
   Blender, Unreal) or `format="abc"` (Alembic: every mesh baked per frame). It lands by the export rule
   (the shot's `characters` folder, or the viewer's own).

6. **Restyle** (humanoids): `action="restyle", prompt="A tired old man walks slowly."` re-does the motion
   already on the character over its keyed range (or `from_frame` / `to_frame`): same path and timing, the
   prompt's style. `strength`: "loose" (most change), "medium" (also keeps hands and feet), "close".
7. **Motion from a film** (humanoids): `action="video_motion", media_path="<film>"` tracks one person (whole
   body in view, still camera) and keys it from `from_frame`. `options={"clean": true}` smooths it with
   Kimodo, `{"in_place": true}` for a moving camera, `film_from` / `film_to` for part of the film.
   `{"via": "sam3d"}` reads the film with ComfyUI's SAM 3D Body nodes (better on hard poses, needs its
   weights in models/detection), `"mediapipe"` with MediaPipe; `"auto"` (default) takes SAM 3D Body when it
   is there. The answer's `read_with` says which read it. `{"fov": 50}` is the camera's vertical field of
   view in degrees, when known.
8. **Lip sync**: `action="lipsync", media_path="<sound or film>"` keys the mouth's blendshapes (ARKit,
   visemes, or one mouth-open shape) from `from_frame`. A character with no mouth shapes cannot: say so.
   It is read from the sound, not word by word - offer to tidy keys by hand.
9. **Layers**: `layer=true` on motion / inbetween / restyle / video_motion puts the take on a new animation
   layer, so the user's keys stay underneath (pose to pose, AI on top). `action="layer_add"` /
   `"layer_list"` make and list layers (`options={"mode": "additive"}`).
10. **Constraints**: `action="constrain", options={"type": "parent", "driven": "Cup", "targets": ["hand.R
    FK"]}` (parent, point, orient, aim, scale); `"list_constraints"`, `"remove_constraint"`
    (`options={"driven": ..}`), `"bake_constraints"` (keys them and removes them).
11. **Pose library**: `"pose_save"` / `"pose_apply"` / `"pose_delete"` with `name`, `"pose_list"`,
    `"flip_pose"` (mirror the whole body). `options={"mirror": true, "blend": 0.5}` on apply.
12. **One pose** ("pose him sitting with arms crossed", "make her kneel", "raise his left arm"):
    `action="pose", from_frame=<frame>, options={"preset": ["sit", "arms_crossed"], "joints": {...}}` keys ONE
    pose on that frame (no motion model needed). Presets are laid over each other in order (`"pose_presets"`
    lists them with the parts and moves): stand, a_pose, t_pose, arms_up, arms_forward, arms_crossed,
    hands_on_hips, hands_behind_head, wave, point, think, sit, sit_floor, kneel, kneel_both, crouch,
    lie_back, lie_front. `joints` adjusts or builds on them, degrees from the T-pose in the parent's frame:
    `{"head": {"turn": 30, "bend": 10}, "upperArm.L": {"down": -60}, "lowerArm.L": {"bend": 45},
    "knees": {"bend": 20}}` (a part without ".L"/".R" is both sides, mirrored; upperArm down 90 = hanging,
    negative = raised; upperLeg forward 90 = thigh level). The body is put on the floor by itself;
    `"hips": {"height": 0.5}` overrides. A pose for each of several frames, then `inbetween`, is
    pose-to-pose animation. With only a `prompt` (Kimodo set up) the pose comes from text instead. The
    answer's `not_understood` lists parts / moves it ignored: fix and retry rather than ignoring it.
    - **Touching the floor** ("a hand on the ground", "a plank", "kneeling on one knee"): `options={"contacts":
      ["foot.L", "hand.R"]}` puts those parts ON the floor, the first one setting the body's height (hand,
      foot or knee, ".L" / ".R"); `{"part": "right hand", "under": true}` puts it straight below its
      shoulder / hip. Elbows and knees bend one way only and every joint stays in a natural range
      (`"limits": false` turns that off - only when the user asks for something a body cannot do).
    - **From a picture** ("pose him like in this photo"): `media_path="<picture of a person>"` reads the
      whole pose off it; contacts and limits still apply. `options={"via": "sam3d" | "mediapipe" | "auto"}`
      as for a film, `"fov"` the picture's vertical field of view, `"facing": "picture"` also turns the
      character the way the person is turned (default: it keeps facing where it faced).

13. **The scene around the character** (`animate_scene(action, options)`; `get_viewer_state()`'s `3d scene`
    line names the shot, cameras, clips and layers). Items, clips, tracks, sources and layers go by name or id.
    - **Clips** (the Clip Sequencer): an AI take becomes a clip at once with `animate_character("motion", ...,
      options={"as_clip": true})`. Keys already in the scene: `"clip_from_keys"` {model or items, name, from, to, keep_keys, track}
      makes a source and a clip of them. `"clip_list"`, `"clip_place"` {source, track?, start}, `"clip_set"`
      {clip, start / length / speed / loop / weight / blend_in / blend_out / mute / name}, `"clip_split"`
      {clip, at}, `"clip_remove"`, `"clip_match"` (line its start up with the clip before), `"clip_bake"`
      (clips back to keys), `"clip_track_add"` / `"clip_track_set"` {track, mute / solo / weight / mode}.
    - **Turntable of each clip**: `render_turntables(clips=None | ["Walk", ...], target="Hero", degrees=360)`
      sets up ONE turntable camera and renders every clip's frames as its own film (`clips=None` = every live
      clip; no clips = the whole shot). `"turntable"` alone only sets the camera up.
    - **Cameras and renders**: `"add_camera"` {name, position, look_at, fov, resolution, active},
      `"set_camera"` {camera, ...}, `"render"` {camera?, clip? or from / to, format mp4 / mov / webm / png,
      shading original / clay / normal / wireframe, name}, `"playblast"` (the same, with the burnt-in HUD).
    - **Keys**: `"key_set"` {item, prop position / rotation / scale / ..., frame, value, ease smooth / linear /
      hold, layer?}, `"key_delete"`, `"key_list"`. `"add_model"` {path} brings a file in.
    - **Layers**: `"layer_set"` {layer, weight / mute / solo / lock / mode}, `"layer_remove"`,
      `"layer_merge_down"`; display layers `"display_layer_add"` {items, name} / `"_set"` / `"_list"`.
    - **Rig extras**: `"ikfk_switch"` / `"ikfk_match"` {control}, `"space_set"` {control, space},
      `"attr_add"` {item, name, value, min, max}, `"sdk_key"` {driver, driven}, `"sdk_remove"`.
    - `"export_usd"` {name} writes the scene (clips and constraints baked) as USD; `"scene_set"` {fps, length,
      active_camera}.

A feature the viewer does not have yet answers "... are not in this viewer yet": pass that on (the viewer
needs updating). Errors to pass on as they are: "AI motion is not set up" / "Auto-rigging is not set up" / "Motion from video is not set up" mean the tools are
not installed on the ComfyUI machine (the viewer's `scripts/rig/install_rig_tools.py`); "no viewer took the
request" means the viewer is not open on the 3D scene. Motion needs a humanoid rig: say so for a creature.
Never claim the motion looks right: the user judges it in the viewer.
