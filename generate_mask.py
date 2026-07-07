from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Load the source image
img = Image.open(r"W:\0193_Never_Stop_Dreaming_Spec\02_build\comfy\sebastian.zilius\output\spec\spec_0030\images\v001\spec_0030_v001_startFrame_00008_.png")
img_array = np.array(img)
height, width = img_array.shape[:2]

print(f"Image size: {width}x{height}")

# Create a mask that's black (protected) for the woman, white (editable) for the background
# Based on the analysis: woman is right-centered, head-and-shoulders
# Approximate bounding box for the subject
left = int(width * 0.35)
top = int(height * 0.05)
right = int(width * 0.95)
bottom = int(height * 0.85)

# Create base elliptical mask
gradient = Image.new('L', (width, height), 0)
draw = ImageDraw.Draw(gradient)

# Draw a filled ellipse in the foreground area
draw.ellipse([left, top, right, bottom], fill=255)

# Feather the edges using Gaussian blur
gradient = gradient.filter(ImageFilter.GaussianBlur(radius=50))

# Invert so black = protected, white = editable
mask = Image.eval(gradient, lambda x: 255 - x)

# Save the mask
mask_path = r"W:\0193_Never_Stop_Dreaming_Spec\02_build\comfy\sebastian.zilius\input\background_mask.png"
mask.save(mask_path)
print(f"Mask saved to {mask_path}")

# For reference, also save a visual guide showing what will be edited
guide_array = np.array(gradient)
guide = Image.new('RGB', (width, height))
pixels = guide.load()
for y in range(height):
    for x in range(width):
        if guide_array[y, x] < 128:
            pixels[x, y] = (255, 0, 0)  # Red = protected (woman)
        else:
            pixels[x, y] = (0, 0, 255)  # Blue = editable (background)

guide_path = r"W:\0193_Never_Stop_Dreaming_Spec\02_build\comfy\sebastian.zilius\output\mask_guide.png"
guide.save(guide_path)
print(f"Mask guide saved to {guide_path}")
