from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from src.utils import format_docs
from src.prompt import prompt
from dotenv import load_dotenv

# from langchain_community.chat_models import ChatOllama

# from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

app = FastAPI()

# LLM
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.2,
    max_tokens=512,
    streaming=True
)
# llm = ChatOllama(model="EEVE-Korean-10.8B:latest")


# Vector Store
db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
retriever = db.as_retriever(search_type="similarity")

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    """user question input model"""
    question: str
    
# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)      

######################################################  Postgres 대화 DB 저장 Start
# from langchain_community.chat_message_histories import (
#     PostgresChatMessageHistory,
# )
# history = PostgresChatMessageHistory(
#     # connection_string="postgresql://postgres:aithe@localhost/chat_history",
#     connection_string="postgresql://postgres:aithe@localhost:5432/postgres",
#     session_id  ="aithe2",
#     # id          = "kevin1",       # TypeError: PostgresChatMessageHistory.__init__() got an unexpected keyword argument 'id'
# )
######################################################  Postgres 대화 DB 저장 End

######################################################  GuardRail Part Start
# from guardrails import Guard
# from guardrails.hub import TwoWords
# guard = Guard().use(TwoWords, on_fail="exception")      # Setup Guard
######################################################  GuardRail Part End

@app.post("/chat/")
async def chat(query: UserQuery):
    """chat endpoint"""
    try:
        # guard.validate(query.question)              ######  GuardRail
        # history.add_user_message(query.question)    #####  Postgres 대화 DB 저장
        
        answer = rag_chain.invoke(query.question).strip()
        
        # guard.validate(guard)                       ######  GuardRail
        # history.add_ai_message(answer)              #####  Postgres 대화 DB 저장
        
        # answer = chain_with_history.invoke(query.question).strip()
        return {"answer": answer}
    except Exception as e:
        print(e)
    