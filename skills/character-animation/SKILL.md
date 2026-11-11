---
name: character-animation
description: Rig and animate a character in the ComfyUI Image Viewer's 3D space - controls on a skinned model, auto-rig a model with no skeleton (UniRig), AI motion from a text prompt and AI in-betweens between keyed poses (NVIDIA Kimodo), feet pinned to the floor, motion restyled ("make the walk sneaky"), motion from a film, lip sync from a sound, AI takes on animation layers, constraints, the pose library, FBX / Alembic export. Use for "make the walk look tired", "restyle this as a march", "copy the dance from this video onto the hero", "lip sync her to this line", "put that on a new layer", "constrain the cup to his right hand", "save this pose", "apply the run pose", "rig this character", "add controls", "auto-rig the creature", "make him walk forward and wave", "animate the hero from frame 0 to 96", "fill in the motion between my two poses", "in-between these keys", "clean up the foot sliding", "export the character as FBX / Alembic".
allowed-tools: animate_character get_viewer_state show_in_viewer
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

A feature the viewer does not have yet answers "... are not in this viewer yet": pass that on (the viewer
needs updating). Errors to pass on as they are: "AI motion is not set up" / "Auto-rigging is not set up" / "Motion from video is not set up" mean the tools are
not installed on the ComfyUI machine (the viewer's `scripts/rig/install_rig_tools.py`); "no viewer took the
request" means the viewer is not open on the 3D scene. Motion needs a humanoid rig: say so for a creature.
Never claim the motion looks right: the user judges it in the viewer.
