# Building on BaseDatasetGenerator

## Overview
The reusable dataset scaffold lives in [data/base.py](data/base.py#L9-L193). Subclass `BaseDatasetGenerator` whenever you need to add a new benchmark dataset loader or prompt generator. The base class handles prompt formatting, optional context stripping, answer extraction, and batched inference bookkeeping so that subclasses only need to focus on preparing dataset rows.

## Extension Checklist
1. Create a new class that inherits from `BaseDatasetGenerator` and call `super().__init__(**kwargs)` to preserve the default prompt setup shown in [data/base.py](data/base.py#L10-L25).
2. Populate `self.dataset` with dictionaries that match the schema defined in [create_dataset](data/base.py#L27-L35). You can assign the list inside `__init__`, inside `create_dataset`, or in a separate loader method.
3. Implement `create_dataset` to fill `self.dataset`. It should return `self.dataset` to make chaining convenient.
4. Gate any additional preprocessing or augmentation behind helper functions so the class stays focused on dataset assembly.

## Dataset Schema
Every row in `self.dataset` must include:
- `question`: full question text, including choices when applicable.
- `question_ill_posed`: the minimal prompt (often the stem without context). When context is embedded in the question, reuse `remove_context` from [data/base.py](data/base.py#L37-L47).
- `answer`: free-form explanation or solution text if you need it for analysis.
- `ref_answer`: the evaluation target (letter for MCQ, numeric string for quantitative tasks).
- Optional fields such as metadata, topic tags, or difficulty can be added; they pass through untouched by the base runner.

## Helper Methods
Leverage the provided utilities instead of re-implementing them:
- `remove_context(question)`: regex-based stem extraction, see [data/base.py](data/base.py#L37-L47).
- `shuffle_choices(choices, correct_answer)`: randomizes options and tracks the correct index, see [data/base.py](data/base.py#L49-L59).
- `extract_answer(content)` and `evaluate(content, ref_answer)`: normalize boxed LaTeX answers, strip commas, and compare numerics as shown in [data/base.py](data/base.py#L61-L133).
- `apply_chat_template(tokenizer, question, thinking=None)`: builds a chat-completion prompt with `<think>` tags, see [data/base.py](data/base.py#L135-L152).
- `run_inference(inference_instance, output_file, begin_index=0, batch_size=32)`: orchestrates batched prompting and appends JSONL results, in [data/base.py](data/base.py#L154-L182).

## Typical Workflow
- Load or generate raw items (CSV, JSON, SQL, etc.).
- Normalize each item into the schema and append to `self.dataset`.
- Optionally call `self.remove_context` or `self.shuffle_choices` while building rows.
- Run `create_dataset` to finalize the list and return it for downstream code.
- Pass the generator instance plus an inference backend to `run_inference` to produce model outputs and store them in a JSONL file.

## Example Skeleton
For a minimal template, study [TestDatasetGenerator](data/base.py#L184-L193). Expand it by reading from your source, populating `self.dataset`, and exposing any dataset-specific knobs (e.g., splits, difficulty filters) through `__init__` arguments forwarded to `super().__init__`.

## Best Practices
- Keep `self.dataset` lightweight; avoid storing large blobs when a file path will do.
- Store intermediate artifacts in the `_data` or `data` subtree to keep git history manageable.
- When adding new helpers, prefer protected methods (prefix with `_`) so the public API stays stable.
- Update documentation and integration tests when you introduce new required fields or evaluation logic.
