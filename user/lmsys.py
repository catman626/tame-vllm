from openai import OpenAI
from datasets import load_dataset
import os

# Login using e.g. `huggingface-cli login` to access this dataset
hf_token=os.environ["HF_TOKEN"]
ds = load_dataset("lmsys/lmsys-chat-1m",  token=hf_token)

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

sampled = ds["train"][0]["conversation"]
chat_response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=sampled
)


print(" >>> extracted Chat response:", chat_response.choices[0].message.content)




