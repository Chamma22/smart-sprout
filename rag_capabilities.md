# RAG Capabilities - What the Agent Can Answer

Based on the Wikipedia corpus in `flower_descriptions.json`. Each page varies in detail, but the following question types are usually well supported across most of the 102 flowers.

---

## Well-Supported Question Types

### Appearance & Identification
- What does [flower] look like?
- What color are [flower] petals?
- How tall does [flower] grow?
- What shape are the leaves or flowers of [flower]?

### Growing Conditions
- What kind of soil does [flower] prefer?
- Does [flower] need full sun or shade?
- When does [flower] bloom?
- What climate does [flower] grow best in?
- How do I plant [flower] bulbs?

### Habitat & Distribution
- Where is [flower] native to?
- What kind of environment does [flower] grow in naturally?

### Propagation
- How do I propagate [flower]?
- Can [flower] be grown from seed?

### Pests & Diseases
- What pests affect [flower]?
- What diseases is [flower] susceptible to?

### Ecology
- What pollinates [flower]?
- Is [flower] good for butterflies or bees?

### Toxicity
- Is [flower] toxic to cats or dogs?
- Is [flower] safe to eat?

### Cultural & Symbolic
- What does [flower] symbolize?
- What is [flower] used for historically?

---

### Similarity & Alternatives
- What flowers look similar to [flower]?
- What other flowers have [color, shape, or trait]?
- What flowers grow well in [condition]?
- Are there alternatives to [flower] that are less toxic?

### Fallback Behavior
- If the agent can't answer a specific question about a flower, it will share what it does know about that flower and offer to find other flowers where that information is available.

---

## Not Well Supported

- **Thinning and spacing advice.** Not covered in the Wikipedia corpus.
- **Side by side species comparisons.** Retrieval may mix chunks from different flowers.
- **Seedling identification from images.** No vision pipeline yet for this task.
- **Real time or location specific advice.** For example, "should I plant today given the weather."
- **Highly specific cultivar questions.** Some cultivars are listed but coverage is inconsistent.
