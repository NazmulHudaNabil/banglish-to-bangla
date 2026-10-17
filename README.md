
---
title: Banglish to Bangla
emoji: 🔤
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Banglish to Bangla

Convert Banglish text into Bangla script using a fine-tuned
Qwen2.5-1.5B-Instruct model with a LoRA adapter.

## Features

- Banglish-to-Bangla text conversion
- FastAPI backend
- Browser-based HTML interface
- JSON API at `/convert`
- API documentation at `/docs`

## API example

POST `/convert`

```json
{
  "text": "ami valo achi"
}
```

Example response:

```json
{
  "input": "ami valo achi",
  "output": "আমি ভালো আছি"
}
```

The response is illustrative. Actual output depends on model behavior.
