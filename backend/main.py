from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings

from src.utils import format_docs
from src.prompt_llamaIndex import prompt

from dotenv import load_dotenv
import os
import chromadb
import json

load_dotenv()

app = FastAPI()

# llm = ChatOllama(model="mistral:latest")
llm = Ollama(model="mistral:latest", temperature=0.1, request_timeout=360000)

# HuggingFaceEmbeddings 초기화
embeddings_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# ChromaDB 클라이언트 생성 및 기존 컬렉션 로드
chroma_client = chromadb.PersistentClient(path="./vector_store")
chroma_collection = chroma_client.get_collection("my_collection")

# ChromaVectorStore 생성
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# StorageContext 생성
storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.embed_model = embeddings_model

# VectorStoreIndex 생성 (이미 존재하는 vector store 사용)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

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

# llama_index
query_engine = index.as_query_engine(
    llm=llm, 
    similarity_top_k=3,
)
query_engine.update_prompts(prompt)

@app.post("/chat/")
async def chat(request: Request):
    """chat endpoint"""
    try:
        body = await request.json()
        query = body["query"]
        answer = query_engine.query(query)
        # answer = rag_chain.invoke(query.question).strip()
        return {"answer": answer}
    except Exception as e:
        print(e)
        return {"error": str(e)}
