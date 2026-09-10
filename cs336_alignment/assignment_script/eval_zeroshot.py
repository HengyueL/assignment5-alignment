"""
Write a script to evaluate OLMo-2-0425-1B performance on GSM8K with zero-shot question_only, zero-shot r1_zero, and few-shot r1_zero_three_shot prompts.

Example usage:

```
uv run python -m cs336_alignment.assignment_script.eval_zeroshot \
    --model-id allenai/OLMo-2-0425-1B \
    --output-dir results/prompting_baselines
```

For each prompt type the script writes
  - {prompt}_generations.jsonl: one record per example (prompt, response, rewards),
  - {prompt}_examples.txt: up to ten readable examples per reward category,
and a shared metrics.json with the reward-category counts for every prompt type.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts"

# Based on Dr. GRPO: stop when the model completes its answer
# https://github.com/sail-sg/understand-r1-zero/blob/
#   c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
R1_ZERO_STOP = ["</answer>"]


@dataclass(frozen=True)
class PromptConfig:
    """A prompt template together with the reward function that parses its outputs."""

    name: str
    template_path: Path
    reward_fn: Callable[..., dict[str, float]]
    stop: list[str] | None

    def load_template(self) -> str:
        return self.template_path.read_text()


PROMPT_CONFIGS: dict[str, PromptConfig] = {
    config.name: config
    for config in [
        PromptConfig(
            name="question_only",
            template_path=PROMPT_DIR / "question_only.prompt",
            # question_only does not ask for <think>/<answer> tags, so it is graded
            # by looking for a final \boxed{} answer instead.
            reward_fn=question_only_reward_fn,
            stop=None,
        ),
        PromptConfig(
            name="r1_zero",
            template_path=PROMPT_DIR / "r1_zero.prompt",
            reward_fn=r1_zero_reward_fn,
            stop=R1_ZERO_STOP,
        ),
        PromptConfig(
            name="r1_zero_three_shot",
            template_path=PROMPT_DIR / "r1_zero_three_shot_gsm8k.prompt",
            reward_fn=r1_zero_reward_fn,
            stop=R1_ZERO_STOP,
        ),
    ]
}

# (1) correct, (2) formatted but wrong, (3) unformatted.
CATEGORIES = {
    "correct": "format_reward=1, answer_reward=1",
    "format_only": "format_reward=1, answer_reward=0",
    "unformatted": "format_reward=0, answer_reward=0",
}


def categorize(reward: dict[str, float]) -> str:
    if reward["answer_reward"] == 1.0:
        return "correct"
    if reward["format_reward"] == 1.0:
        return "format_only"
    return "unformatted"


def load_gsm8k(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read GSM8K jsonl, splitting the `{rationale} #### {answer}` field into its answer."""
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                {
                    "question": record["question"],
                    "ground_truth": record["answer"].split("####")[-1].strip(),
                    "reference_solution": record["answer"],
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def generate(
    server: VLLMServer,
    prompts: list[str],
    sampling_params: dict,
    batch_size: int | None,
) -> list[VLLMCompletion]:
    chunk_size = batch_size or len(prompts)
    completions: list[VLLMCompletion] = []
    start_time = time.monotonic()
    for start in range(0, len(prompts), chunk_size):
        completions.extend(server.generate_completions(prompts[start : start + chunk_size], sampling_params))
        logger.info(
            "Generated %d/%d responses (%.1fs elapsed).",
            len(completions),
            len(prompts),
            time.monotonic() - start_time,
        )
    return completions


def evaluate_prompt(
    server: VLLMServer,
    config: PromptConfig,
    examples: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    template = config.load_template()
    prompts = [template.format(question=example["question"]) for example in examples]

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": 1,
        "seed": args.seed,
    }
    if config.stop is not None:
        sampling_params["stop"] = config.stop
        sampling_params["include_stop_str_in_output"] = True

    logger.info("[%s] Generating %d responses.", config.name, len(prompts))
    completions = generate(server, prompts, sampling_params, args.batch_size)
    assert len(completions) == len(prompts)

    logger.info("[%s] Grading %d responses.", config.name, len(completions))
    records = []
    for example, prompt, completion in zip(examples, prompts, completions):
        reward = config.reward_fn(completion.text, example["ground_truth"])
        records.append(
            {
                "prompt_type": config.name,
                "prompt": prompt,
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "num_response_tokens": len(completion.token_ids),
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"],
                "category": categorize(reward),
            }
        )

    num_examples = len(records)
    counts = {category: sum(record["category"] == category for record in records) for category in CATEGORIES}
    metrics = {
        "prompt_type": config.name,
        "num_examples": num_examples,
        "accuracy": sum(record["answer_reward"] for record in records) / num_examples,
        "format_rate": sum(record["format_reward"] for record in records) / num_examples,
        "avg_reward": sum(record["reward"] for record in records) / num_examples,
        "counts": counts,
        "num_truncated": sum(record["finish_reason"] == "length" for record in records),
        "avg_response_tokens": sum(record["num_response_tokens"] for record in records) / num_examples,
    }
    return metrics, records


def write_generations(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_examples(path: Path, records: list[dict], num_per_category: int) -> None:
    """Dump a few readable prompt/response pairs per reward category for the writeup."""
    with path.open("w") as f:
        for category, description in CATEGORIES.items():
            matching = [record for record in records if record["category"] == category][:num_per_category]
            f.write(f"{'=' * 80}\nCATEGORY: {category} ({description}) — showing {len(matching)}\n{'=' * 80}\n")
            for index, record in enumerate(matching):
                f.write(
                    f"\n--- example {index} ---\n"
                    f"PROMPT:\n{record['prompt']}\n"
                    f"RESPONSE:\n{record['response']}\n"
                    f"GROUND TRUTH: {record['ground_truth']}\n"
                    f"FINISH REASON: {record['finish_reason']}\n"
                )
            f.write("\n")


def summarize(all_metrics: list[dict]) -> str:
    header = f"{'prompt':<20}{'n':>7}{'accuracy':>10}{'format':>10}{'correct':>10}{'fmt-only':>10}{'unfmt':>10}{'trunc':>8}"
    lines = [header, "-" * len(header)]
    for metrics in all_metrics:
        counts = metrics["counts"]
        lines.append(
            f"{metrics['prompt_type']:<20}{metrics['num_examples']:>7}"
            f"{metrics['accuracy']:>10.3f}{metrics['format_rate']:>10.3f}"
            f"{counts['correct']:>10}{counts['format_only']:>10}{counts['unformatted']:>10}"
            f"{metrics['num_truncated']:>8}"
        )
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "gsm8k" / "test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "prompting_baselines")
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=list(PROMPT_CONFIGS),
        choices=list(PROMPT_CONFIGS),
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256, help="Prompts per vLLM request.")
    parser.add_argument("--num-examples-per-category", type=int, default=10)
    # vLLM server options.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--no-launch-server",
        dest="launch_server",
        action="store_false",
        help="Attach to a vLLM server that is already running at --host:--port.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    examples = load_gsm8k(args.data_path, args.limit)
    logger.info("Loaded %d examples from %s.", len(examples), args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    server = VLLMServer(
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        gpu=args.gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        launch_server=args.launch_server,
    )
    logger.info("Starting vLLM server for %s.", args.model_id)
    server.start()

    all_metrics = []
    try:
        for prompt_type in args.prompt_types:
            config = PROMPT_CONFIGS[prompt_type]
            metrics, records = evaluate_prompt(server, config, examples, args)
            all_metrics.append(metrics)

            write_generations(args.output_dir / f"{prompt_type}_generations.jsonl", records)
            write_examples(
                args.output_dir / f"{prompt_type}_examples.txt",
                records,
                args.num_examples_per_category,
            )
            logger.info("[%s] %s", prompt_type, json.dumps(metrics))
    finally:
        server.stop()

    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    print(summarize(all_metrics))
    print(f"\nWrote generations, examples, and metrics to {args.output_dir}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(make_parser().parse_args())
