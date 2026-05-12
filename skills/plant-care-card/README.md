# plant-care-card skill

An Anthropic style skill for the Smart Sprout agent. Given the name of a flower covered by Smart Sprout's 102 flower Wikipedia corpus, the skill instructs the agent to assemble a one page care card covering description, growing conditions, toxicity, pollinators, pests, and cultural notes.

## Files

- `SKILL.md`. Frontmatter (name, description) plus the agent facing instructions that drive when and how the skill runs.
- `template.md`. The care card layout with `{{placeholder}}` fields. The skill instructs the agent to fill these placeholders using only information returned by the `plant_lookup` tool.
- `README.md`. This file.

## How to use

The skill is invoked by the Smart Sprout agent (`src/agent.py`) when a user asks for a "care card," "summary card," "cheat sheet," or one page overview of a flower. The agent has access to the `plant_lookup` tool, which queries the ChromaDB vector store built from `data/flower_descriptions.json`. The skill prescribes the sequence of `plant_lookup` calls needed to fill every field, then tells the agent to render the completed template directly in chat.

Example user prompts:

- "Make a care card for the sunflower."
- "Give me a cheat sheet on foxglove."
- "Quick reference on water lily, please."

## Why a skill

The same information could be produced by asking the agent freeform, but the result would vary every time. The skill enforces a single layout, a fixed set of fields, and a strict no fabrication rule. Any field the corpus is silent on must be marked "Not in corpus" rather than filled from the LLM's prior knowledge. That predictability is the value the skill adds on top of the existing RAG tool.
