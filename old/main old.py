
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import *
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
import pdfplumber
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import json
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio


CHAT_HISTORY_DIR = "./chat_histories"

if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)



## extract text from pdfs
def extract_text_from_pdfs(directory):
    data = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if the file is a PDF
        if filepath.endswith('.pdf'):
            print(f'Processing {filename}...')
            
            # Open the PDF file
            with pdfplumber.open(filepath) as pdf:
                # Extract text from each page (you can modify this based on your needs)
                for page in pdf.pages:
                    text = page.extract_text()
                    # Do something with the extracted text (print it, store it, etc.)
                    data.append(text)
            print(f'{filename} processed.\n')
    return data\

##clean html doc
def clean_document(docs):

    for doc in docs:
        cleaned_content = ' '.join(doc.page_content.split())
        #print(cleaned_content)
        doc.page_content = cleaned_content




##download html doc
loader = WebBaseLoader(["https://promtior.ai/","https://promtior.ai/solutions/"])

docs = loader.load()

clean_document(docs)



## load pdf doc
pdf_data = extract_text_from_pdfs("./")
pdf_data = [Document(page_content=content) for content in pdf_data]
clean_document(pdf_data)

for doc in pdf_data:
    docs.append(doc)

embeddings = OllamaEmbeddings(model="phi3:mini")


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)



llm = Ollama(model="phi3:mini", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


retriever = vector.as_retriever()



prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)





# Define FastAPI app and request/response models
class InputModel(BaseModel):
    input: str

class OutputModel(BaseModel):
    output: str

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

@app.post("/agent", response_model=OutputModel)
async def process_request(input_model: InputModel):
    try:
        async def event_stream():
            # Start the processing in a separate task
            result = retrieval_chain.invoke({"input": input_model.input})["answer"]

            for chunk in result:
                yield f"data: {chunk}\n\n"

        # Return the streaming response immediately
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)