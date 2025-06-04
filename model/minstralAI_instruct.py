from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM, pipeline
from datasets import load_dataset 

model_name = "microsoft/phi-2"
#mistralai/Mistral-7B-Instruct-v0.3
token = "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"

dataset= load_dataset("multi_woz_v22")
# Load tokenizer and model with the token for authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Create the pipeline with model and tokenizer objects (no use_auth_token here)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# If you want to test simple prompt generation:
prompt = (
    "You are a helpful assistant. Respond to the user's questions in plain English only. "
    "Do not include code or technical syntax in your responses.\n\n"
    "respond formally to the user's question.\n"
)
output = pipe(prompt, max_length=100, truncation=True)


prompt = input("Prompt: ")
output = pipe(prompt, max_length=100, truncation=True)
print(output[0]['generated_text'])
