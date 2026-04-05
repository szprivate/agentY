---
name: image-batch
description: Use whenever a job requires more than one execution run — "enhance all 10 photos", "make 5 variations of each image", "try different styles/CFG/prompts", "turn these frames into a video", or any brief where output count > 1. The Researcher sets the iteration plan; this skill tells the Brain what to change per run.
allowed-tools:
---

# image-batch

Take over the iteration count `count_iter` from the Researcher. 

Use `patch_workflow` to apply the specified changes per run, decide which parameters to change. 

Instead of creating one single workflow, create an array of workflows for each run, and forward this array to the executor. 

IMPORTANT 
For the prompt, make sure that you're creating a separate, distinct prompt for every workow - if for example the user asks to create 9 variations of an input image, create 9 prompt that are different fron wach other. NEVER create one prompt that describes all 9 images.