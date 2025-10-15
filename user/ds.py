from datasets import load_dataset
import os

# Login using e.g. `huggingface-cli login` to access this dataset
hf_token=os.environ["HF_TOKEN"]
ds = load_dataset("lmsys/lmsys-chat-1m", token=hf_token)

print(ds['train'][0]["conversation"])


