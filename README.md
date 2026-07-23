# ML Template "Flexit" (Don't Confuse It with Flexi)
Flexit was originally intended to be a private machine learning framework. However, the project became larger than expected, so I've decided to make it public.
If you're interested in building an AI model from scratch, you can use this project as a learning resource or as a starting template for your own architecture.
This ML Template uses the Transformer, the very same architecture ChatGPT, Gemini, Claude, DeepSeek, Qwen, GLM, Gemma, GPT-OSS, and other models being hosted.
# Features
- Generic training pipeline
- Generic benchmarking framework
- Generic evaluation framework
- Designed to be adaptable to different model architectures
- JSONL dataset support
- History filtering instead of Git LFS, keeping the repository lightweight
## Where do I start..?
Install the dependencies:
`pip install -r requirements.txt`
Now, run the tests with your data/data.jsonl. [How do I format my dataset?](#how-do-i-format-my-dataset)

After that, you can build your own model architecture on top of the framework and train it using the included trainer.
## Unit Tests?
Basically, the project includes unit tests covering many components, including:
- Inference
- Attention
- Transformer blocks
- Embeddings
- Token encoding
- Dataset loading
- Data collators
- Training
- Evaluation
- Benchmarking
- Loss functions
- Metrics
- Optimizers
- Schedulers
- Sampling
- Model configuration
- The example Flex model
The tests are intentionally simple. They exist to verify that the framework works correctly while remaining easy to understand.
## How do I format my dataset?
Datasets should be stored in the JSONL format. The example model (`flux/`)
Place your training dataset in `data/data.jsonl`.
If the `data/` directory doesn't exist yet, simply create it.
## Why is the test model called Flex?
This repository is a template, and templates need an example implementation. The included example model is named Flex, and many of the tests reference it for demonstration purposes.
Feel free to rename or replace it with your own architecture.
"Call this template Flex. Let the Transformer flex its power."
