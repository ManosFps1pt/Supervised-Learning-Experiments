import torch
from transformers import pipeline, GenerationConfig

MODEL_ID = "google/gemma-3-4b-it"  # use gemma-3-1b-it if your machine struggles

pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 1000
config.do_sample = False

while True:
    messages = [
        {
            "role": "user",
            "content": input("input: ")
        }
    ]

    out = pipe(messages, return_full_text=False, generation_config=config)
    print(out[0]["generated_text"])