# GSM8K Pilot: Qwen2.5-1.5B + DataSmith DPO

This is a preliminary benchmark artifact for the DataSmith launch/article work. It records the
successful run, the failed pilot attempts, and enough reproduction detail for another agent or human
to continue from the same setup.

## Result

Fixed eval set: GSM8K test indices `0-99`.

Scorer: exact final numeric answer match. The scorer prefers the number after `####`; if missing, it
uses the last number in the response.

| Model | Training data | Training method | Correct | Accuracy |
|---|---|---:|---:|---:|
| `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | none | none | 34 / 100 | 34% |
| Qwen2.5-1.5B + DataSmith LoRA | 68 filtered DataSmith preference pairs | DPO, 80 steps | 43 / 100 | 43% |

Delta: `+9` points absolute on this fixed 100-example GSM8K subset.

## Why This Run Matters

The useful signal came from DataSmith's weak-vs-strong byproduct:

- `prompt`: generated math problem
- `chosen`: strong-solver answer
- `rejected`: weak-solver answer

This maps directly to DPO preference training. SFT on the same style of data did not improve broad
GSM8K in the pilot runs.

## Generated Data

Kaggle notebook:

- `datasmith-gsm8k-benchmark`
- URL: `https://www.kaggle.com/code/atharvakanherkara/datasmith-gsm8k-benchmark`

Kaggle dataset:

- `datasmith-math-78-dpo`
- URL: `https://www.kaggle.com/datasets/atharvakanherkara/datasmith-math-78-dpo`
- File: `datasmith_math_78_dpo.jsonl`

Local source files from the generation run:

- `runs/datasmith_gsm8k_openai_100/accepted.jsonl`
- `runs/datasmith_gsm8k_openai_100/rejected.jsonl`
- `runs/datasmith_gsm8k_openai_100/datasmith_math_78_dpo.jsonl`
- `runs/datasmith_gsm8k_openai_100/datasmith_math_78_dpo_conversational.jsonl`

These files are not committed because `runs/` is ignored.

## Generation Setup

DataSmith accepted examples were generated from real GSM8K train seeds.

Seed source:

- Hugging Face dataset: `openai/gsm8k`, split `train`
- Seed count: 25 examples
- Seeds were used as grounding examples, not as direct training data.

Generation roles:

| Role | Model / path |
|---|---|
| Challenger | `gpt-4.1-mini` |
| Weak solver | `naive-weak-math` local solver |
| Strong solver | `gpt-5.4-mini` |
| Judge | `gpt-5.4-mini` |

The local weak solver intentionally made shallow arithmetic guesses. This created a clean
weak-vs-strong separation and made the accepted examples useful as preference pairs.

Generation result:

```json
{
  "accepted": 78,
  "rejected": 44,
  "attempts": 122,
  "target_count": 100,
  "target_met": false,
  "models": {
    "challenger": "gpt-4.1-mini",
    "weak_solver": "naive-weak-math",
    "strong_solver": "gpt-5.4-mini",
    "judge": "gpt-5.4-mini"
  }
}
```

## DPO Data Filtering

The raw DPO export had 78 records. Filtering removed rows with obvious inconsistency or ambiguity.

Filter result:

```text
before: 78
after: 68
```

Filtered out rows where prompt/chosen text or metadata tags contained:

- `inconsistent`
- `impossible`
- `cannot be determined`
- `not enough information`
- `no unique`
- `as stated`
- metadata tag `inconsistent`
- metadata tag `ambiguous`

## Fine-Tuning Setup

Environment:

- Kaggle notebook
- Free GPU: Tesla T4
- Unsloth

Base model:

- `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

DPO config:

```python
DPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=80,
    learning_rate=5e-5,
    beta=0.1,
    fp16=True,
    logging_steps=10,
    output_dir="datasmith_gsm8k_dpo_lora",
    optim="adamw_8bit",
    seed=13,
    report_to="none",
    max_length=1024,
    max_prompt_length=512,
)
```

Training summary:

```text
Num examples = 68
Total steps = 80
Trainable parameters = 18,464,768 / 1,562,179,072 (1.18%)
Training loss = 0.0633
Runtime = 204s
```

## Eval Harness

Eval set creation:

```python
from datasets import load_dataset

ds = load_dataset("gsm8k", "main")
test = ds["test"]
eval_rows = []

for i in range(100):
    row = test[i]
    eval_rows.append({
        "id": f"gsm8k-test-{i}",
        "index": i,
        "question": row["question"],
        "answer": row["answer"],
        "gold_number": row["answer"].split("####")[-1].strip().replace(",", ""),
    })
```

Scorer:

```python
import re

def extract_number(text):
    if "####" in text:
        tail = text.split("####")[-1]
        nums = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
        if nums:
            return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def is_correct(pred_text, gold_number):
    return extract_number(pred_text) == str(gold_number).replace(",", "").strip()
```

Prompt:

```text
Solve this grade school math problem. Show concise reasoning and end with #### <number>.

Problem:
{question}

Answer:
```

## Failed / Neutral Pilot Runs

These are important for the article because they explain why DPO is the better fit for DataSmith.

### Gemma Baseline Attempt

Model:

- `unsloth/gemma-2-2b-it-bnb-4bit`

Result:

- Model loaded successfully after accepting gated access.
- In the Kaggle/Unsloth path, generation produced only whitespace for both chat-template and plain
  prompts.
- The run was abandoned and replaced with Qwen2.5-1.5B.

### SFT On Raw DataSmith ChatML

Training data:

- 78 DataSmith SFT rows exported as ChatML
- 40 SFT steps

Result:

```text
Baseline: 34 / 100
SFT:      34 / 100
Delta:    0
```

Issue:

- The first export had JSON strings inside chat messages.
- Several accepted examples contained noisy/inconsistent reasoning.

### Clean-Filtered SFT

Training data:

- 62 filtered clean SFT rows
- 100 SFT steps

Partial result:

- At 70 eval examples, only 11 were correct.
- The run was stopped because it was clearly regressing.

Likely cause:

- SFT overfit a tiny synthetic set and hurt broad GSM8K behavior.
- DataSmith's accepted examples are better used as preference data than direct SFT labels.

## Article Angle

Suggested framing:

> DataSmith generated preference data as a byproduct of validation. On a fixed GSM8K subset,
> DPO-tuning Qwen2.5-1.5B on 68 filtered DataSmith preference pairs improved exact-match accuracy
> from 34% to 43%.

Emphasize:

- This is a small pilot, not a benchmark paper.
- The gain came from DPO, not SFT.
- The weak-vs-strong loop naturally creates `chosen` / `rejected` pairs.
- Failed SFT pilots support the product thesis: DataSmith's differentiated output is preference
  data, not generic synthetic SFT data.

## Follow-Up Work

- Run three seeds for DPO training to check variance.
- Generate 200-500 cleaner accepted DPO pairs.
- Publish the Kaggle notebook and predictions.
- Add a README benchmark table only after the notebook is public.
- Consider a dedicated DataSmith hidden-constraint eval in addition to broad GSM8K.
