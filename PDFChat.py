# This Python file uses the following encoding: utf-8
import os
import streamlit as st
import time

import tempfile
import lancedb   # ✅ DuckDB 기반 LanceDB 연결 라이브러리
from langchain_community.vectorstores import LanceDB

from loguru import logger
from dotenv import load_dotenv

# LangChain 관련
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint  

# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================
# Hugging Face 토큰 설정
# =========================
def set_hf_token():
    hf_token = None

    # 1. Streamlit Cloud 환경 (Secrets 사용)
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    # 2. 로컬 개발 환경 (.env 사용)
    else:
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        st.error("❌ HuggingFace API Token을 찾을 수 없습니다. Secrets 탭이나 .env에 설정하세요.")
        st.stop()

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    return hf_token


# =========================
# Main 함수
# =========================
def main():
    st.set_page_config(page_title="PDF/문서 AI Q&A", page_icon="📄", layout="wide")
    st.header("📄 문서 업로드 & AI Q&A")

    # API Token 설정
    hf_token = set_hf_token()

    # 문서 업로드
    uploaded_file = st.file_uploader("문서를 업로드하세요 (PDF, DOCX, PPTX 지원)", type=["pdf", "docx", "pptx"])

    if uploaded_file:
        with st.spinner("📑 문서 처리 중..."):
            documents = load_document(uploaded_file)
            if documents is None:
                return

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # 임베딩 생성
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            #vectorstore = FAISS.from_documents(docs, embeddings)

            # vectorstore = Chroma.from_documents(docs, embeddings)

            db = lancedb.connect("duckdb_storage")  # DuckDB 파일 기반 DB
            table_name = "docs"

            if table_name in db.table_names():
                vectorstore = LanceDB(connection=db, table_name=table_name, embedding=embeddings)
                vectorstore.add_documents(docs)
            else:
                vectorstore = LanceDB.from_documents(docs, embeddings, connection=db, table_name=table_name)

            # 대화 체인 생성
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("✅ 문서 처리 완료! 이제 질문할 수 있습니다.")

    # Q&A 영역
    if "conversation" in st.session_state:
        query = st.text_input("질문을 입력하세요:")
        if query:
            with st.spinner("🤖 AI가 생각 중..."):
                result = safe_query(st.session_state.conversation, query)
                response = result["answer"]

                st.markdown(f"**답변:** {response}")

                if "source_documents" in result:
                    with st.expander("📂 참조 문서 보기"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**문서 {i+1}:** {doc.page_content[:500]}...")


# =========================
# 안전한 질의 함수
# =========================
# def safe_query(chain, query, max_retries=3):
#     for attempt in range(max_retries):
#         try:
#             return chain({"question": query})
#         except Exception as e:
#             wait_time = 5 * (attempt + 1)
#             st.warning(f"⚠️ API 에러 발생: {e}\n{wait_time}초 후 재시도합니다...")
#             time.sleep(wait_time)
#     raise Exception("❌ API 호출 실패 - 잠시 후 다시 시도해주세요.")

def safe_query(chain, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke({"question": query})   # ✅ invoke 로 교체
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            st.warning(f"⚠️ API 에러 발생: {e}\n{wait_time}초 후 재시도합니다...")
            time.sleep(wait_time)
    raise Exception("❌ API 호출 실패 - 잠시 후 다시 시도해주세요.")

# =========================
# 대화 체인 생성
# =========================
def get_conversation_chain(vectorstore):
    # HuggingFaceHub LLM 불러오기 (Inference API)
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",   # 무료 권장 모델
        model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr"),
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain


# =========================
# 문서 로드 함수
# =========================
# def load_document(uploaded_file):
#    name, ext = os.path.splitext(uploaded_file.name.lower())

#   if ext == ".pdf":
#        loader = PyPDFLoader(uploaded_file)
#    elif ext in [".docx", ".doc"]:
#        loader = Docx2txtLoader(uploaded_file)
#    elif ext in [".pptx", ".ppt"]:
#        loader = UnstructuredPowerPointLoader(uploaded_file)
#    else:
#        st.error("지원하지 않는 파일 형식입니다. (pdf, docx, pptx 지원)")
#        return None
#
#    return loader.load()


# import tempfile

def load_document(uploaded_file):
    name, ext = os.path.splitext(uploaded_file.name.lower())

    # UploadedFile -> temp file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(tmp_path)
    elif ext in [".pptx", ".ppt"]:
        loader = UnstructuredPowerPointLoader(tmp_path)
    else:
        st.error("지원하지 않는 파일 형식입니다. (pdf, docx, pptx 지원)")
        return None

    return loader.load()


# =========================
if __name__ == '__main__':
    main()