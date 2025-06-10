
# from langchain_community.llms import Ollama
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
from pathlib import Path
from pprint import pprint
question = "1"
#Loader
file_path = Path("../data")
raw_data = json.loads((file_path / "calendar_data.json").read_text())
# Convert raw data to Document objects
data = [Document(page_content=str(entry)) for entry in raw_data]
#Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
#Embedding + vector store
embeddings = OllamaEmbeddings(model = "llama3")
vector_store = InMemoryVectorStore(embedding = embeddings)
vector_store.add_documents(all_splits)

model = OllamaLLM( model = "llama3")

template = """
You are a helpful assistant who responds politely.

Here are some relevant facts:
{list}

here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
while question != "0":
    #User question
    question = input("What is your question? (type '0' to exit): ")
    if question == "0":
        break

    #Run similarity search
    retrieved_docs = vector_store.similarity_search(question, k=4)
    relevant_facts = [doc.page_content for doc in retrieved_docs]

    #Run RAG chain
    result = chain.invoke({
        "list": "\n\n".join(relevant_facts),
        "question": question
    })
    print(result)
# #User question
# question = "What is the best time to workout?"

# #Run similarity search
# retrieved_docs = vector_store.similarity_search(question, k=4)
# relevant_facts = [doc.page_content for doc in retrieved_docs]

# #Run RAG chain
# result = chain.invoke({
#     "list": "\n\n".join(relevant_facts),
#     "question": question
# })
# print(result)