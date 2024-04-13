from transformers import pipeline

pipe = pipeline("text-generation", model="georgesung/llama2_7b_chat_uncensored")
prompt_text = ("hi ")
generated_text = pipe(prompt_text, num_return_sequences=1)

print(generated_text[0]['generated_text'])