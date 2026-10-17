---
language:
  - bn
  - en
license: mit
base_model: unsloth/Qwen2.5-1.5B-Instruct
pipeline_tag: text2text-generation
tags:
  - banglish
  - bangla
  - bengali
  - transliteration
  - unsloth
  - peft
  - lora
  - qlora
  - text-generation
---

# Banglish-to-Bangla — Qwen2.5-1.5B LoRA Fine-Tuned Model

A fine-tuned language model designed to convert **Banglish (Bangla written using Latin characters) into Bangla script**.

This model is based on Qwen2.5-1.5B-Instruct and fine-tuned using Unsloth with parameter-efficient fine-tuning (LoRA/QLoRA).

## Model Overview

| Property | Details |
|---|---|
| Task | Banglish-to-Bangla transliteration/conversion |
| Base model | `unsloth/Qwen2.5-1.5B-Instruct` |
| Model architecture | Qwen2.5-1.5B-Instruct |
| Fine-tuning framework | Unsloth |
| Fine-tuning method | LoRA / QLoRA |
| Adapter format | PEFT |
| Model type | LoRA adapter |
| Primary output language | Bangla (Bengali script) |
| Intended use | Banglish text conversion |

**Note:** This repository contains the fine-tuned adapter, not the complete base model. The base model must also be loaded for inference.

## What Is Banglish?

Banglish is Bangla written using the Latin alphabet instead of the Bengali script.

For example:

- Banglish: `ami valo achi`
- Bangla: `আমি ভালো আছি`

The spelling of Banglish is not standardized. The same Bangla sentence may be written in multiple ways using Latin characters. The model aims to learn the conversion patterns represented in its training data.

## Example

**Input**

```text
ami valo achi
```

**Expected output**

```text
আমি ভালো আছি
```

This is an illustrative example, not a claim of measured model performance. Actual outputs may vary.

## Intended Use

This model is intended for:

- Converting Banglish text into Bangla script.
- Experimenting with Bangla language technology.
- Building Banglish-to-Bangla text conversion applications.
- Research and educational projects involving Bengali NLP.
- Integrating into a backend API or a web application.

## Training Details

The model was fine-tuned using Unsloth and a parameter-efficient adaptation approach.

The training configuration included:

- Training epochs: 1
- Per-device training batch size: 2
- Gradient accumulation steps: 8
- Learning rate: 2e-4
- Learning-rate scheduler: cosine
- Warmup ratio: 0.03
- Maximum sequence length: 512, if this matches the final training configuration
- Evaluation metric used for checkpoint selection: evaluation loss
- Checkpoint selection: best checkpoint based on evaluation loss

The adapter was saved separately from the original base model.

### Dataset

The model was trained on a dataset prepared for Banglish-to-Bangla conversion.

Dataset details:

- Dataset source: **https://huggingface.co/datasets/aplycaebous/BanglaTLit/**
- Training examples: **10_000**
- Validation examples: **1500**
- Test examples: **2500**

Please fill in these details before publishing. Ensure that the dataset license permits your intended use and redistribution.

## Evaluation

Formal evaluation results have not yet been reported.

For a reliable evaluation, assess the model on a held-out test dataset that was not used for training or model selection.

Recommended metrics:

| Metric | Purpose |
|---|---|
| Character Error Rate (CER) | Measures character-level conversion errors |
| Word Error Rate (WER) | Measures word-level errors |
| Exact Match Accuracy | Measures the proportion of predictions exactly matching the reference |
| Manual evaluation | Assesses spelling, meaning preservation, and naturalness |

Evaluation results will be added after testing.

## How to Use

### Requirements

Install the necessary libraries:

```bash
pip install torch transformers peft accelerate
```

### Load the model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "nazmulhudanabil/banglish-to-bangla-qwen2.5-1.5b"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_REPO,
)

model.eval()
```

Replace `YOUR_USERNAME/banglish-to-bangla-qwen2.5-1.5b` with the actual Hugging Face repository ID of this adapter.

### Generate a conversion

```python
text = "ami valo achi"

prompt = f"""Convert the following Banglish text into natural Bangla script.
Preserve the original meaning.

Banglish: {text}
Bangla:"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(result)
```

The inference prompt should match the prompt format used during training as closely as possible.

## Limitations

- Output quality depends on the training dataset and the spelling variations in the input.
- Ambiguous Banglish text may have multiple valid Bangla interpretations.
- The model may occasionally generate spelling mistakes, alter the intended meaning, or produce unrelated text.
- Performance on domains and dialects not represented in the training data is unknown.
- The model has not been assigned a verified quantitative performance score in this model card.

## License

The adapter's license and usage rights depend on the applicable base-model license, dataset licenses, and the terms under which the adapter was created.

Verify the licenses before publishing. Do not claim that this adapter is Apache-2.0 licensed unless you have confirmed that this license is appropriate for the adapter and its training data.

## Acknowledgements

- [Qwen2.5](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Author

**Author:** MD Nazmul Huda Nabil

**Model repository:** https://huggingface.co/nazmulhudanabil/banglish-to-bangla-qwen2.5-1.5b

**Project source code:** https://github.com/NazmulHudaNabil/banglish-to-bangla
