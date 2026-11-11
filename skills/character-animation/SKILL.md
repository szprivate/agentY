---
name: character-animation
description: Rig and animate a character in the ComfyUI Image Viewer's 3D space - controls on a skinned model, auto-rig a model with no skeleton (UniRig), AI motion from a text prompt and AI in-betweens between keyed poses (NVIDIA Kimodo), feet pinned to the floor, FBX / Alembic export. Use for "rig this character", "add controls", "auto-rig the creature", "make him walk forward and wave", "animate the hero from frame 0 to 96", "fill in the motion between my two poses", "in-between these keys", "clean up the foot sliding", "export the character as FBX / Alembic".
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

Errors to pass on as they are: "AI motion is not set up" / "Auto-rigging is not set up" mean the tools are
not installed on the ComfyUI machine (the viewer's `scripts/rig/install_rig_tools.py`); "no viewer took the
request" means the viewer is not open on the 3D scene. Motion needs a humanoid rig: say so for a creature.
Never claim the motion looks right: the user judges it in the viewer.
