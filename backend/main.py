from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings

from src.utils import format_docs
from src.prompt import prompt

from dotenv import load_dotenv
import os
import chromadb

load_dotenv()
app = FastAPI()

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = Ollama(
    model="mistral:latest",
    base_url=ollama_base_url,
    temperature=0.1,
    request_timeout=360000
)

# HuggingFaceEmbedding 초기화
embeddings_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")        

# ChromaDB 클라이언트 생성 및 기존 컬렉션 로드
chroma_client = chromadb.PersistentClient(path="./vector_store")
chroma_collection = chroma_client.get_collection("my_collection")

# ChromaVectorStore 생성
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# StorageContext 생성
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# LlamaIndex Settings에 임베딩 모델 설정
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
    
# LlamaIndex Query Engine 설정
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
)
query_engine.update_prompts(prompt)

@app.post("/chat/")
async def chat(user_query: UserQuery): # Request 대신 Pydantic 모델 직접 사용
    """chat endpoint"""
    try:
        # body = await request.json() # Pydantic 모델 사용 시 불필요
        # query = body["query"]      # Pydantic 모델 사용 시 불필요

        answer = query_engine.query(user_query.query) # user_query.question 대신 user_query.query 사용

        return {"answer": str(answer)} # LlamaIndex Response 객체를 문자열로 변환
    except Exception as e:
        print(e)
        # 에러 발생 시 클라이언트에 에러 메시지를 반환하는 것이 좋습니다.
        return {"error": str(e)}, 500 # HTTP 500 상태 코드와 함께 에러 반환
    