
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

import json
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from apiModels import *
from datetime import datetime, timezone, timedelta
import time

CHAT_HISTORY_DIR = "./chat_histories"

if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)
OLLAMA_HOST = os.getenv('OLLAMA_HOST')


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

embeddings = OllamaEmbeddings(model="phi3:mini",base_url=OLLAMA_HOST)


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)



llm = Ollama(base_url=OLLAMA_HOST,model="phi3:mini", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


retriever = vector.as_retriever()



prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided, if you dont know, answer with "I'm sorry I dont know"
context: 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)





# Define FastAPI app and request/response models

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

app.debug = True

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/api/tags", response_model=ModelsResponse)
async def models():
    models = json.load(open("./models.json"))

    # Return a list of available models
    return ModelsResponse(models=models)

@app.get("/api/version", response_model=VersionResponse)
async def models():
    # Return a list of available models
    return VersionResponse(version="0.2.1")

@app.post("/api/chat", response_class=StreamingResponse)
async def process_request(request: Request):
    body = await request.json()
    print(body)  # Log the raw input
    chat_request = ChatRequest(**body)

#async def chat(request: ChatRequest):
   
    try:
        async def event_stream():
            start_time = time.time()
            model_name = "phi3:mini"
            prompt_eval_count = 0
            prompt_eval_duration = 0
            eval_count = 0
            eval_duration = 0
            async for event in retrieval_chain.astream({"input": chat_request.messages[-1].content}):
                answer = event.get("answer", "")
                prompt_eval_count += 1
                prompt_eval_duration += event.get("prompt_eval_duration", 0)
                eval_count += 1
                eval_duration += event.get("eval_duration", 0)

                now = datetime.now(timezone(timedelta(hours=-3)))
                formatted_time = now.isoformat()

                data = {
                    "model": model_name,
                    "created_at": formatted_time,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                        "images": None
                    },
                    "done": False
                }
                yield f"{json.dumps(data)}\n"

                total_duration = time.time() - start_time
            data = {
                "model": model_name,
                "created_at": datetime.now(timezone(timedelta(hours=-3))).isoformat(),
                "done": True,
                "total_duration": total_duration * 1000000000,
                "load_duration": 0,
                "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_count": eval_count,
                "eval_duration": eval_duration
            }
            yield f"{json.dumps(data)}\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#@app.post("/api/chat", response_class=StreamingResponse)
#async def process_request(request: Request):
#    body = await request.json()
#    print(body)  # Log the raw input
#    chat_request = ChatRequest(**body)
#
##async def chat(request: ChatRequest):
#   
#    try:
#        async def event_stream():
#            start_time = time.time()
#            model_name = "phi3:mini"
#            prompt_eval_count = 0
#            prompt_eval_duration = 0
#            eval_count = 0
#            eval_duration = 0
#
#            async for event in retrieval_chain.astream({"input": chat_request.messages[0].content}):
#                answer = event.get("answer", "")
#                prompt_eval_count += 1
#                prompt_eval_duration += event.get("prompt_eval_duration", 0)
#                eval_count += 1
#                eval_duration += event.get("eval_duration", 0)
#
#                now = datetime.now(timezone(timedelta(hours=-3)))
#                formatted_time = now.isoformat()
#
#                data = {
#                    "model": model_name,
#                    "created_at": formatted_time,
#                    "message": {
#                        "role": "assistant",
#                        "content": answer,
#                        "images": None
#                    },
#                    "done": False
#                }
#                yield f"data: {json.dumps(data)}\n\n"
#
#                total_duration = time.time() - start_time
#            data = {
#                "model": model_name,
#                "created_at": datetime.now(timezone(timedelta(hours=-3))).isoformat(),
#                "done": True,
#                "total_duration": total_duration * 1000000000,
#                "load_duration": 0,
#                "prompt_eval_count": prompt_eval_count,
#                "prompt_eval_duration": prompt_eval_duration,
#                "eval_count": eval_count,
#                "eval_duration": eval_duration
#            }
#            yield f"data: {json.dumps(data)}\n\n"
#
#        return StreamingResponse(event_stream(), media_type="text/event-stream")
#
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
#

@app.post("/agent", response_model=OutputModel)
async def process_request(input_model: InputModel):
    try:
        async def event_stream():
            # Start the processing in a separate task
            events = []
            async for event in retrieval_chain.astream({"input": input_model.input}):
                events.append(event)
                print(event.get("answer", ""))

            for event in events:
                yield f"data: {event.get('answer', '')}\n\n"

        # Return the streaming response immediately
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    