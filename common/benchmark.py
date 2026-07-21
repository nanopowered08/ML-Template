from __future__ import annotations

from typing import Iterable


class Benchmark:
    """
    Generic benchmark for every Fl-x model.

    Measures:

    • Loss
    • Perplexity
    • Human score (optional)

    Uses only the generic Generator and Evaluator.
    """

    def __init__(
        self,
        generator,
        evaluator,
    ):

        self.generator = generator
        self.evaluator = evaluator


    def run(
        self,
        dataloader,
        prompts: Iterable[str],
        human_eval: bool = False,
    ) -> dict:

        metrics = self.evaluator.evaluate(
            dataloader
        )

        ratings = []

        print("=" * 60)
        print("Benchmark")
        print("=" * 60)

        for prompt in prompts:

            print()
            print("-" * 60)
            print("Prompt:")
            print(prompt)
            print()

            tokens = self.generator.encode(
                prompt
            )

            encodedTensors = self.generator.generate(
                tokens
            )

            responseEncoded = encodedTensors.squeeze(0).tolist()

            response = self.generator.decode(
                responseEncoded
            )

            print("Response:")
            print(response)
            print()

            if human_eval:

                while True:

                    score = input(
                        "Rating (1-5, Enter to skip): "
                    ).strip()

                    if score == "":
                        break

                    try:

                        score = int(score)

                        if 1 <= score <= 5:

                            ratings.append(score)
                            break

                    except ValueError:
                        pass

                    print(
                        "Please enter a number from 1 to 5."
                    )

        if ratings:

            metrics["human_score"] = (
                sum(ratings)
                / len(ratings)
            )

            metrics["rated_samples"] = len(
                ratings
            )

        else:

            metrics["human_score"] = None
            metrics["rated_samples"] = 0

        return metrics
