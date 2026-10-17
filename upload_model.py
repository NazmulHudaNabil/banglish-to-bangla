
from huggingface_hub import HfApi

REPO_ID = "nazmulhudanabil/banglish-to-bangla-qwen2.5-1.5b"
ADAPTER_PATH = "./final_adapter"

api = HfApi()

# Create the repository if it does not already exist
api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,
)

# Upload the complete adapter folder
api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"Adapter uploaded: https://huggingface.co/{REPO_ID}")
