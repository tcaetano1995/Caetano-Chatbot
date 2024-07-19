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
from langchain.chains import LLMChain
from langchain.chains.combine_documents import *
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
import pdfplumber
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import json
from langchain_core.runnables.history import RunnableWithMessageHistory

CHAT_HISTORY_DIR = "./chat_histories"

if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)


def save_message(session_id: str, role: str, content: str):
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    
    # Load existing messages or create a new session
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            session_data = json.load(f)
    else:
        session_data = {"session_id": session_id, "messages": []}

    session_data["messages"].append({"role": role, "content": content})

    # Save the updated session data back to JSON
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=4)

def load_session_history(session_id: str) -> BaseChatMessageHistory:
    session_file = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    chat_history = ChatMessageHistory()

    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            session_data = json.load(f)
            for message in session_data["messages"]:
                chat_history.add_message({"role": message["role"], "content": message["content"]})

    return chat_history

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]


def invoke_and_save(session_id, input_text):
    save_message(session_id, "human", input_text)
    
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)
    return result


def get_context_from_documents(retrieved_docs):
    return "\n".join([doc.page_content for doc in retrieved_docs])


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



### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store = {} 


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
     verbose=False
)


question = ""
chat_history = []
while question != "exit":
    print()
    question = input("Ask a question: ")
    if question == "exit":
        break
    answer =invoke_and_save("1234", question)
