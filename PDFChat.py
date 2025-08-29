# This Python file uses the following encoding: utf-8
import os, sys
import streamlit as st
import tiktoken
import time

from loguru import logger
from dotenv import load_dotenv

# Cloud에서는 st.secrets 사용
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# HuggingFaceHub에 자동 적용
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# LangChain 관련
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================
# Main 함수
# =========================
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF/문서 AI Q&A", page_icon="📄", layout="wide")

    st.header("📄 문서 업로드 & AI Q&A")

    # 업로드
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
            vectorstore = FAISS.from_documents(docs, embeddings)

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
def safe_query(chain, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain({"question": query})
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
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",   # 권장 무료 모델
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
def load_document(uploaded_file):
    name, ext = os.path.splitext(uploaded_file.name.lower())

    if ext == ".pdf":
        loader = PyPDFLoader(uploaded_file)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(uploaded_file)
    elif ext in [".pptx", ".ppt"]:
        loader = UnstructuredPowerPointLoader(uploaded_file)
    else:
        st.error("지원하지 않는 파일 형식입니다. (pdf, docx, pptx 지원)")
        return None

    return loader.load()


# =========================
if __name__ == '__main__':
    main()
