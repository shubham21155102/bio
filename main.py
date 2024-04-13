# from transformers import pipeline

# pipe = pipeline("text-generation", model="georgesung/llama2_7b_chat_uncensored")

# model = AutoModelForCausalLM.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
# prompt_text = ("hi ")
# generated_text = pipe(prompt_text, num_return_sequences=1)

# print(generated_text[0]['generated_text'])
from transformers import AutoTokenizer, LlamaForCausalLM
import os
os.environ["HF_TOKEN"] = ""
cache_dir = "/workspace/bio/models/"
# os.makedirs(cache_dir, exist_ok=True)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=30)
decoded_output =tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(decoded_output)