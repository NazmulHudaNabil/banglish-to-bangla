
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./final_adapter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()



def convert_banglish_to_bangla(text: str) -> str:
    prompt = f"""Convert the following Banglish text into natural Bangla script.
Preserve the original meaning.

Banglish: {text}
Bangla:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Return only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


if __name__ == "__main__":
    text = input("Enter Banglish: ")
    result = convert_banglish_to_bangla(text)
    print("Bangla:", result)
