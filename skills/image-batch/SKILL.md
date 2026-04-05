---
name: image-batch
description: Use whenever a job requires more than one execution run — "enhance all 10 photos", "make 5 variations of each image", "try different styles/CFG/prompts", "turn these frames into a video", or any brief where output count > 1. The Researcher sets the iteration plan; this skill tells the Brain what to change per run.
allowed-tools:
---

# image-batch

Take over 

Take over the iteration count `count_iter` from the Researcher. Use `patch_workflow` to apply the specified changes per run, decide which parameters to change. Instead of creating single workflow, create an array of workflows for each run, and forward this array to the executor. For the prompt, make sure that you're creating 9 separate prompts for 9 separate images - not one prompt for 9 images.