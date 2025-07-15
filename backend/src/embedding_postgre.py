import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from llama_index.core.schema import TextNode, Document  # postgre vector store
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
from llama_index.vector_stores.postgres import PGVectorStore
import pandas as pd
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv(verbose=True)

class ExcelReader:
    """엑셀 파일을 읽어서 Document 객체로 변환하는 클래스"""

    def __init__(self):
        pass

    def load_data(self, file, extra_info=None):
        df = pd.read_excel(file)
        docs = []
        for _, row in df.iterrows():
            metadata = {
                'keyword': self.handle_nan(row['keyword']),
                'page': self.handle_nan(row['page']),
                'plaintiff': self.handle_nan(row['plaintiff']),
                'defendant': self.handle_nan(row['defendant']),
                'another': self.handle_nan(row['another']),
                'money': self.handle_nan(row['money']),
                'unit': self.handle_nan(row['unit']),
                'etc': self.handle_nan(row['etc']),
                'interest_rate': self.handle_nan(row['interest rate']),
                'time_start': self.format_date(row['time_start']),
                'time_end': self.format_date(row['time_end']),
                'time_start_txt': self.handle_nan(row['time_start']),
                'time_end_txt': self.handle_nan(row['time_end'])
            }
            if extra_info:
                metadata.update(extra_info)

            # 문서 내용 생성 및 Document 객체로 변환
            content = f"{self.handle_nan(row['plaintiff'])}\n" \
                      f"{self.handle_nan(row['defendant'])}\n" \
                      f"{self.handle_nan(row['another'])}\n" \
                      f"{self.handle_nan(row['money'])}\n" \
                      f"{self.handle_nan(row['unit'])}\n" \
                      f"{self.handle_nan(row['etc'])}\n" \
                      f"{self.handle_nan(row['text'])}"
            doc = Document(text=content.strip(), metadata=metadata)
            docs.append(doc)
        return docs

    def handle_nan(self, value):
        if pd.isna(value) or value is None:
            return ''
        elif isinstance(value, float) and np.isnan(value):
            return ''
        return str(value)

    def format_date(self, value):
        if pd.isna(value) or value is None:
            return ''
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')
        try:
            parsed_date = pd.to_datetime(value)
            return parsed_date.strftime('%Y-%m-%d')
        except Exception:
            return str(value)

    def format_date(self, date_value):
        if isinstance(date_value, datetime):
            return date_value.isoformat()
        elif pd.isna(date_value):
            return ''
        else:
            return str(date_value)


def load_files(input_dir):
    file_paths = []
    for ext in [".pdf", ".xlsx", ".xls"]:  # 처리하고자 하는 파일 확장자
        file_paths.extend(list(input_dir.glob(f"*{ext}")))

    if not file_paths:
        raise ValueError(f"No supported files found in {input_dir}")

    print(f"Found files: {[f.name for f in file_paths]}")

    parser = LlamaParse(
        api_key="llx-JItT6ZbUs6c05fS0nNr3luAD13gxvfPouCrnwmNbZlv2nblg",
        result_type="markdown",  # or "text"
        verbose=True,
    )

    excel_reader = ExcelReader()
    file_extractor = {
        ".pdf": parser,
        ".xlsx": excel_reader,
        ".xls": excel_reader
    }

    reader = SimpleDirectoryReader(
        input_files=[str(p) for p in file_paths],
        file_extractor=file_extractor
    )
    docs = reader.load_data()
    return docs


def split(docs):
    """문서를 일정 크기의 청크로 분할"""
    indexing = SentenceSplitter(chunk_size=512, chunk_overlap=0)
    nodes = indexing.get_nodes_from_documents(docs)
    # Document 객체들을 TextNode 객체로 변환하며 분할
    return nodes


def is_doc(obj):
    if isinstance(obj, Document):  # vector_index - PDF 파일 대상
        return True
    elif isinstance(obj, TextNode):  # vector_index - 일반 텍스트 대상
        return False
    return None

def create_index(docs, schema_name="public", table_name="tmp"):
    """PostgreSQL 벡터 저장소 생성 및 문서 인덱싱"""
    vector_store = PGVectorStore.from_params(
        database="chatbot",
        host="localhost",
        password="chatbot01",
        port="5432",
        user="chatbot01",
        schema_name=schema_name,
        table_name=table_name,
        embed_dim=1024,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="dragonkue/BGE-m3-ko")
    # embed_model = OpenAIEmbedding()

    doc_or_node = is_doc(docs[0])
    if doc_or_node is None:
        raise ValueError("Invalid document type for indexing.")

    try:
        if doc_or_node:
            # PDF 기반 Document 객체 인덱싱
            index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                show_progress=True,
                embed_model=embed_model
            )
        else:
            # TextNode 기반 인덱싱
            index = VectorStoreIndex(
                docs,
                storage_context=storage_context,
                show_progress=True,
                embed_model=embed_model
            )
        return index

    except Exception as e:
        print("create_index Exception:", str(e))
        return None


if __name__ == "__main__":
    try:
        file_path = Path("../data").resolve()
        docs = load_files(file_path)
        nodes = split(docs)
        index = create_index(nodes, schema_name="public", table_name="tmp_chatbot")

        if index is not None:
            index.storage_context.persist()  # 벡터 스토어에 저장
            print("Embedding PostgreSQL Success")
        else:
            print("Index creation failed.")

    except Exception as e:
        print("Embedding Postgre create_vector_store Exception:", str(e))
