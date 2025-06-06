
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM( model = "llama3")

template = """
You are a helpful assistant who responds politely.

Here are some relevant facts" {list}

here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

result = chain.invoke({"list": [], "question": "What is the weather like today and where am I?"})

print(result)