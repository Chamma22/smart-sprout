---
name: plant-care-card
description: Generate a structured care card for a flower using Smart Sprout's RAG corpus. Use this whenever the user asks for a "care card," "summary card," "cheat sheet," or a one page overview of a specific flower. Call the plant_lookup tool to gather information, then fill the template in template.md and present the completed card to the user.
---

# Plant Care Card

This skill turns Smart Sprout's free form RAG output into a consistent, scannable care card for a single flower.

## When to use

Use this skill when the user asks for any of:

- "Make a care card for [flower]"
- "Summary card / cheat sheet / quick reference for [flower]"
- "Give me a one pager on [flower]"
- "What do I need to know about [flower]?" (when they want a structured summary rather than a conversational answer)

Don't use this skill for free form questions like "is foxglove toxic?". Answer those directly with `plant_lookup`. The care card is for when the user wants the full structured profile.

## How to run the skill

1. **Identify the flower.** If the user names the flower explicitly, use that name. If they reference an earlier identification ("the one I just showed you", "that one"), resolve the reference from conversation memory before continuing. If the flower is ambiguous, ask one clarifying question before proceeding.

2. **Gather information.** Call the `plant_lookup` tool with each of these queries in sequence. Don't skip any, since every field on the card depends on these calls.

    - `'{flower} description appearance'`
    - `'{flower} native range habitat'`
    - `'{flower} growing conditions sun water soil'`
    - `'{flower} toxicity pets humans'`
    - `'{flower} pollinators wildlife'`
    - `'{flower} pests diseases problems'`
    - `'{flower} cultural significance uses'`

   If a query returns nothing useful, mark that field on the card as **"Not in corpus."** Don't invent or guess.

3. **Read the template.** Open `template.md` (in this skill's folder). It's a markdown care card with placeholders in `{{double_braces}}`.

4. **Fill the template.** Replace each placeholder using only information from the `plant_lookup` results. Keep entries short, one or two sentences per field. Use bullet points where the template uses them. If you have multiple pieces of information for one field, pick the most relevant one or two.

5. **Present the card.** Output the completed markdown directly in the chat. Don't wrap it in code fences. Don't add commentary before or after the card unless the user asked a follow up question alongside the request.

## Output rules

- **No fabrication.** Every claim on the card must trace back to a `plant_lookup` result. If the corpus is silent on a field, write "Not in corpus."
- **No outside knowledge.** Even if you know a fact about the flower from training data, don't include it unless `plant_lookup` returned it. The point of the card is to summarize what the project's corpus contains.
- **Keep it scannable.** No paragraph in any field should exceed three sentences. Bullet lists should have at most four items.
- **Common name in the title, scientific name below.** If the corpus gives a binomial like *Helianthus annuus*, include it. Otherwise omit that line rather than guessing.

## Failure modes

- If `plant_lookup` returns "No relevant plant information found" for every query, don't produce a card. Tell the user the flower isn't in the corpus and offer to look up a related one.
- If the user asks for a care card on a flower the agent just misidentified, prefer the corrected name (from the most recent `record_correction` or from the user's stated correction).
