# ML-Template "Flexit" (don't mistake it with Flexi)
This was originally supposed to be a private framework, but the work is too much and so I'm making it public. For those who want to start a new AI model from scratch, take your time learning from this one or use the template to train one.
## Features
- Generic trainer
- Generic Benchmarker
- Should be mostly compatible with any other models built from a different architecture
- Generic Evaluator
- Datasets support for JSON, yay!
- History being overwritten by a filter so it goes through without Github asking for LFS instead.
## How to Run
Just do 'pip install -r requirements.txt'. From there, make your ML framework on top and train using the generic trainer.
## Unit Tests?
Yes, there are tests. In my time making this, those were just simple, simple tests. They tested inference, attention, benchmark, Transformer block, collator, dataset loading, test embeddings, encoding, evaluating, the Flex model's configuration, inference, loss, metrics, the model itself, the optimizer, sampling, scheduling, training, and the other tests.
## What dataset do I use?
Well, the dataset should be formatted as a JSONL, as my original purpose for this framework was making a model that could code, but is a SLM.
To use your dataset,  place it in `data/` and the name of the training data should be `data.jsonl`. (if the data folder hasn't been created yet, create it.)
## Why is the model name Flex and the unit tests are pointing to it?
Well this is a template. Of course I need a name for it.
"Call this template Flex. Let the Transformer "Flex" it's power."
