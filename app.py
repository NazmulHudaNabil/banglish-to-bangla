
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Configuration
# -------------------------
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "nazmulhudanabil/banglish-to-bangla-qwen2.5-1.5b"

APP_DIR = Path(__file__).resolve().parent
HF_TOKEN = os.getenv("HF_TOKEN") or None

app = FastAPI(
    title="Banglish-to-Bangla API",
    description="Convert Banglish text into Bangla script using a fine-tuned Qwen model.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=APP_DIR), name="static")

# -------------------------
# Load model once at startup
# -------------------------
tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_REPO,
        token=HF_TOKEN,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPO,
        token=HF_TOKEN,
    )

    model.eval()
    

# -------------------------
# Request and response
# -------------------------
class ConvertRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class ConvertResponse(BaseModel):
    input: str
    output: str


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return FileResponse(APP_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/convert", response_model=ConvertResponse)
def convert(request: ConvertRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Please enter Banglish text.")

    # IMPORTANT:
    # Keep this prompt consistent with the template used during training.
    prompt = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context.\n"
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Convert the Banglish text into Bangla script while preserving its meaning.\n\n"
        f"### Input:\n{text}\n\n"
        "### Response:\n"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return ConvertResponse(input=text, output=result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {type(exc).__name__}",
        ) from exc
