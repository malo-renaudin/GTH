from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "gpt2"  # or the full snapshot folder path from above
tok = AutoTokenizer.from_pretrained(model_path, cache_dir="scripts/train/.cache", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="scripts/train/.cache", local_files_only=True)
model.eval()
model.config.use_cache = True

inputs = tok("Hello world", return_tensors="pt")
out = model(**inputs, use_cache=True)
print("use_cache in config:", model.config.use_cache)
print("past_key_values is", type(out.past_key_values))
if out.past_key_values:
    for i, layer in enumerate(out.past_key_values):
        print(i, [None if t is None else t.shape for t in layer])
else:
    print("No KV cache returned")