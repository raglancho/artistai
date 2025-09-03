# This Python file uses the following encoding: utf-8
import os
import streamlit as st
import time
import tempfile

from dotenv import load_dotenv

# LangChain 관련
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import LanceDB
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)


# =========================
# This Python file uses the following encoding: utf-8
import os
import streamlit as st
import tempfile
import time
from loguru import logger
from dotenv import load_dotenv

# LangChain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint

# LanceDB
import lancedb
from langchain_community.vectorstores import LanceDB


# =========================
# Hugging Face Token
# =========================
def set_hf_token():
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    else:
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        st.error("❌ HuggingFace API Token을 찾을 수 없습니다.")
        st.stop()

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    return hf_token


# =========================
# 문서 로드
# =========================
def load_document(uploaded_file):
    name, ext = os.path.splitext(uploaded_file.name.lower())

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
# 안전한 질의
# =========================
def safe_query(chain, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke({"question": query})
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            st.warning(f"⚠️ API 에러 발생: {e}\n{wait_time}초 후 재시도...")
            time.sleep(wait_time)
    raise Exception("❌ API 호출 실패 - 잠시 후 다시 시도해주세요.")


# =========================
# 대화 체인
# =========================
def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # 무료 추천 모델
        temperature=0.3,
        max_new_tokens=512,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr"),
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="문서 기반 Q&A", page_icon="📄", layout="wide")
    st.header("📄 문서 업로드 & AI Q&A")

    hf_token = set_hf_token()

    # LanceDB 경로 (Streamlit Cloud에서도 사용 가능)
    db = lancedb.connect("lancedb_data")

    # 문서 업로드
    uploaded_file = st.file_uploader("문서를 업로드하세요 (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"])

    if uploaded_file:
        with st.spinner("📑 문서 처리 중..."):
            documents = load_document(uploaded_file)
            if documents is None:
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            if not docs:
                st.error("❌ 문서에서 텍스트를 추출하지 못했습니다. 다른 파일을 업로드해주세요.")
                return

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 🔹 기존 테이블이 있으면 추가, 없으면 새로 생성
            if "docs" in db.table_names():
                table = db.open_table("docs")
                table.add(docs)
            else:
                table = LanceDB.from_documents(docs, embeddings, connection=db, table_name="docs")


            # table = db.open_table("docs") if "docs" in db.table_names() else db.create_table("docs", data=None)
            vectorstore = LanceDB.from_documents(docs, embeddings, connection=db, table_name="docs")

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
if __name__ == "__main__":
    main()
