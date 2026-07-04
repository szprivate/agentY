---
name: workflow-template-analysis
description: Analyze the relationship between workflow_templates.json catalog and workflow_recipes.json technical database in agentY project
source: auto-skill
extracted_at: '2026-07-04T20:26:42.854Z'
---

# Workflow Template Analysis Skill

This skill helps analyze the relationship between the workflow template catalog and the recipe database in the agentY project architecture.

## Key Files to Examine

1. `agenty_core/config/workflow_templates.json` - High-level template catalog
2. `agenty_core/config/workflow_recipes.json` - Detailed technical implementation database
3. `agentY/config/settings.json` - Project configuration referencing templates
4. Template files in `workflow_templates` repository

## Analysis Procedure

1. First examine `agenty_core/config/workflow_templates.json`:
   - This file contains a simple mapping of template names to descriptions
   - Used for template discovery and user-facing template selection
   - Contains both API and local templates with descriptive text

2. Then examine `agenty_core/config/workflow_recipes.json`:
   - This file contains detailed technical information organized as task → model → node clusters
   - Provides implementation blueprints for workflow assembly
   - Contains information about required nodes, connection patterns, and boundary ports

3. Compare the two files:
   - Check if all templates in workflow_templates.json have corresponding entries in the recipe database
   - Identify redundancy or complementary purposes
   - Understand how they work together in the agent pipeline

## What to Look For

1. Templates that exist in the catalog but might be missing from the recipe database
2. Discrepancies in template descriptions between the two sources
3. Template naming conventions and how they map between the catalog and implementation
4. Technical details in recipes that aren't captured in the high-level catalog

## Common Analysis Questions

- Is workflow_templates.json redundant given we have detailed recipes?
- Does the catalog contain all templates from the recipe database?
- How do they work together in template selection and workflow assembly?
- Are there templates that exist in one but not the other?
- What information is lost if we only had one or the other?