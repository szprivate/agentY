---
name: model-from-reference
description: Model a building, object or set in a 3D app (Blender through its MCP) so that it matches a reference photo — the way a 3D artist does it - measure the photo, match the camera, block out grey shapes until the silhouette and proportions match, THEN details, THEN materials and light. Use for "model this house / object / room in Blender from this photo", "rebuild this in 3D", "make a 3D version of this picture", "match this reference in Blender".
allowed-tools: blender__execute_blender_code blender__render_viewport_to_path blender__get_objects_summary blender__get_screenshot_of_window_as_image analyze_image get_image_resolution run_script write_text_file read_text_file show_in_panel
---

# Model from a reference photo

The order is the whole method. Never build geometry and materials in one go: when the
proportions are wrong, every material, every detail and every light has to be redone
with them. Work in stages, and do not leave a stage until its check passes.

| stage | what you do | leave it when |
|---|---|---|
| 1 Measure | read positions off the photo, turn them into metres | a dimension table exists |
| 2 Camera | match the photo's camera before any geometry | horizon and frame match |
| 3 Blockout | plain grey boxes and slabs, no detail, no colour | the silhouette check passes |
| 4 Detail | windows, railings, bracing, trim, props | the detail check passes |
| 5 Look | materials, sun, sky, final render | colours and light match |

## One build script, rebuilt every pass

Keep the whole model in ONE Python file, `output/scripts/<name>_build.py` (write it with
`write_text_file`). At its top, every dimension is a named constant with a comment saying
where it came from (`DECK_Z = 3.3  # deck top: 0.22 of image height above the sand line`).
The script first deletes its own collection, then builds everything from the constants.

Run it with `blender__execute_blender_code`:

```python
p = r"<abs path>/output/scripts/<name>_build.py"
exec(compile(open(p, encoding="utf-8").read(), p, "exec"))
```

A correction is a changed constant and a re-run: the scene is rebuilt clean every time.
Never patch the live scene object by object over many calls; leftovers pile up (stray
blocks, doubled parts) and you lose track of what is really there.

## 1 Measure the photo

1. `get_image_resolution` on the reference; note its aspect ratio.
2. Make a measuring copy with `run_script`: draw a labelled grid every 5 % of width and
   height over the photo (PIL), save it as `<name>_grid.jpg`.
3. Look at the grid copy yourself (`analyze_image(mode='full')`) and read positions as
   fractions of the image: ground contact line, deck/floor line, eaves, ridge, top of mast,
   left and right extents of each part, the horizon line.
4. Pick ONE scale anchor of known size: a door (~2.0 m), a railing (~1.0 m), a step
   (~0.18 m), a person (~1.75 m), a storey (~2.7 m). Pixels between two lines at the same
   depth divided by the anchor's pixels gives metres.
5. Write the table into the script as constants. Note which parts face the camera, which
   recede, and which project toward it (a wing in front, a balcony on that wing...).

## 2 Match the camera first

- Resolution: the photo's aspect (portrait stays portrait).
- Keep the camera level (no tilt, no roll) and move the frame with lens shift instead:
  a level camera puts the horizon at the frame centre; for a portrait frame
  `shift_y = horizon_from_top - 0.5`, for landscape multiply that by height/width.
  Positive `shift_y` moves the frame UP, so the subject moves DOWN in the picture.
- Camera height: whatever sits ON the horizon line in the photo is at eye height. If the
  horizon cuts the stilts 1.5 m above the sand, the camera is at 1.5 m.
- Distance: `d = real_width / (2 * tan(hfov / 2) * width_fraction)`. Start at 50 mm.
- Aim with `(target - cam.location).to_track_quat('-Z', 'Y')`. Hand-written Euler angles
  are how a camera ends up rolled, or pointing backwards at an empty frame.
- Which side of the subject is the camera on? A wall you can see means the camera is on
  that wall's side. Check this against the photo before moving on.

## 3 Blockout: grey only

Simple boxes and slabs for every mass: legs, deck, each wing, each roof plane, mast.
ONE flat grey material, no detail. Render fast (Eevee or Workbench, ~800 px).

Gable roofs: build each roof plane from its four vertices (eave left, eave right, ridge
right, ridge left) so the ridge is the highest edge. A rotated slab overshoots its ridge
and turns the roof into an inverted V.

**The silhouette check, by numbers, not by eye.** Project key points into the camera with
`bpy_extras.object_utils.world_to_camera_view(scene, cam, point)` and compare the
(x, 1 - y) fractions with your measurements: ground line, deck line, ridge, mast top, left
and right extents. Every one within ~0.03 of the photo before you go on. This check cannot
hallucinate, so it decides; the picture comparison below finds what the numbers missed.

Then compare pictures: `analyze_image(file_paths=[reference, render], mode='full')` when
your model can see images, otherwise mode `describe`. Ask open questions ("List the
differences in shape, proportions and camera, most important first"), never a checklist
to confirm. If a comparison says something is missing or extra, check it in the scene
(bounding boxes in camera view), not by object names.

## 4 Detail

Windows and doors where the photo has them (count them), railings, bracing, stairs,
trim, props. Re-run the silhouette check; details must not move the masses.

## 5 Look

- Materials last. Take colours from the photo and convert sRGB to linear before putting
  them in a Principled BSDF (`c / 12.92` below 0.04045, else `((c + 0.055) / 1.055) ** 2.4`).
- Use the "Standard" view transform for colours that match the photo. AgX and Filmic wash
  out saturated colours such as a blue sky or orange wood.
- Sun direction from the photo's shadows. The side facing the camera is usually lit.
- Final render: Cycles, GPU, 64-128 samples with denoising is plenty. Save as `<name>_final.png`.

## Report

Show the final render next to the photo. List what still differs honestly; do not rate
your own match ("6/10"), and do not claim a measurement you did not check.
