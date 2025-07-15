from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, PromptTemplate, Settings, QueryBundle, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from src.utils import format_docs
from src.prompt_llamaIndex import prompt

from dotenv import load_dotenv
import os
import chromadb
import json

from typing import List, Optional
# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS for frontend
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

# LLM 설정
# 로컬 환경
# llm = Ollama(model="mistral:latest", temperature=0.1, request_timeout=360000)
# 도커 환경
llm = Ollama(model="mistral:latest", base_url="http://ollama_dev:11434", temperature=0.1, request_timeout=360000)

# HuggingFace 임베딩 모델
embed_model = HuggingFaceEmbedding(model_name="dragonkue/BGE-m3-ko")

# 벡터 저장소 연결
vector_store = PGVectorStore.from_params(
    database="chatbot",
# 로컬 환경
    # host="localhost",
# 도커 환경
    host="host.docker.internal",
    password="chatbot01",
    port="5432",
    user="chatbot01",
    schema_name="public",
    table_name="tmp_chatbot",
    embed_dim=1024,
)

# 사용자 쿼리 모델
class UserQuery(BaseModel):
    question: str


# 사용자 정의 후처리기
class CustomPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        print("========== Custom Postprocessor ========== ")
        for n in nodes:
            print(f"file_name: {n.metadata.get('file_name', 'N/A')}")
            print(n)
        print("Query Bundle:", query_bundle)
        return nodes


# 후처리기 목록
custom_postprocessor = CustomPostprocessor()
node_postprocessors = [
    custom_postprocessor,
    MetadataReplacementPostProcessor(target_metadata_key="window")
]

# StorageContext 생성
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 인덱스 로딩
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

# 쿼리 엔진 구성
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=10,
    node_postprocessors=node_postprocessors
)

# 프롬프트 업데이트
query_engine.update_prompts(prompt)


@app.post("/chat/")
async def chat(request: Request):
    """chat endpoint"""
    try:
        body = await request.json()
        query = body["query"]

        answer = query_engine.query(query)
        return {"answer": answer}
    except Exception as e:
        print(e)
        return {"error": str(e)}
