# Flexit

*(not to be confused with "Flexi".. idk if that exists)*

Flexit is supposed to be a ML Template based on the Transformer that powers *′* ChatGPT, Gemini, Claude, DeepSeek, Qwen, GLM, Gemma, and GPT-OSS.

*′* Note: Im talking about the architecture powering those not the ML Template powering it.

THIS started as a private template cuz i wanna make a model for myself, but it GOT SO big + grew into something worth sharing. ID LET YOU use it as a **learning resource** to understand how modern LLMs are built or as a **starting template** for your own model architecture.

---

## ANDDD WE HAVE-

- A generic training pipeline
- AND a generic benchmarking framework
- YEP EVERYTHING IS GENERIC, so also generic evaluation framework
- Adaptable to different model architectures cuz LOOK AT THE LAST 3 POINTS.
- JSONL dataset support ONLY.
- History got rewritten by the filter erasing that 300MB test model

---

## Where do I start?

**1. Install the deps**
```bash
pip install -r requirements.txt
```

**2. Format the dataset.**

Place your training data at `data/data.jsonl` (create the `data/` folder if it doesn't exist). [How do I format my dataset?](#-how-do-i-format-my-dataset)

**3. TEST the thing.**

Verify everything works with the included test suite (see [the unit tests](#-unit-tests-literally)).

**4. Build your model**

Use the `flex/` example model as a reference, then build and train your own architecture on top of the framework using the included trainer.

---

## Unit Tests... LITERALLY.

The project ships with tests covering the core framework components:

| Category | Covers |
|---|---|
| Model internals | Attention, Transformer blocks, Embeddings, Token encoding |
| Data | Dataset loading, Data collators |
| Training | Training loop, Loss functions, Optimizers, Schedulers |
| Evaluation | Evaluation, Benchmarking, Metrics, Sampling |
| Config & Inference | Model configuration, Inference, the example Flex model |

Tests are intentionally simple CUZ-
They exist to confirm the framework behaves correctly, while staying easy to read and learn from.

---

## How do I format my dataset?

Datasets use the **JSONL** format, with each line following the **ChatML** structure:

```jsonl
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" },
    { "role": "assistant", "content": "Hi there! How can I help you today?" }
  ]
}
```

Place your file at `data/data.jsonl`.
Trust me.

---

## SO why "Flex"?

Every template needs a working example.. SO-
This one's called **Flex**, and most of the tests reference it for demonstration. Rename it, gut it, or replace it entirely with your own architecture.

Also my brain was thinking:

> Call this template Flex. Let the Transformer flex its power.
