# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM ,MistralForCausalLM, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
token = "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"

pipe = pipeline("text-generation", model=model_name, use_auth_token=token)


messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

# Load model directly


# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

model = MistralForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


#Generate text
generator = pipeline('text-generation', model=model_name, tokenizer=tokenizer)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_text= generator(prompt,max_length = 50)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generate_text[0]['generated_text'])