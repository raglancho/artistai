# This Python file uses the following encoding: utf-8
import os, sys
import streamlit as st
import tiktoken
import time
import openai



from loguru import logger

#from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

#from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import Docx2txtLoader
#from langchain.document_loaders import UnstructuredPowerPointLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.memory import ConversationBufferMemory
#from langchain.vectorstores import FAISS

# 기존 langchain -> langchain_community 로 교체
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# langchain core & openai 최신 버전
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

'''
def main():
    from dotenv import load_dotenv
    load_dotenv()

    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "Hello? Do you have a any Question ?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("Write a your question !!"):
    
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

         #   with st.spinner("Thinking..."):
         #       result = chain({"question": query})
         #       with get_openai_callback() as cb:
         #           st.session_state.chat_history = result['chat_history']
         #       response = result['answer']
         #       source_documents = result['source_documents']

        with st.spinner("Thinking..."):
            result = safe_query(st.session_state.conversation, query)
            response = result["result"]   # RetrievalQA에서는 result 키가 다름
            
            if "source_documents" in result:
                with st.expander("Check sources"):
                    for doc in result["source_documents"][:3]:
                        st.markdown(doc.metadata["source"])
                        st.caption(doc.page_content[:200] + "...")
                        st.markdown(response)

                with st.expander("Check your Document!!"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

'''

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

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 
        
        with open(file_name, "wb") as file:  # doc.name 
        
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        
            if '.pdf' in doc.name:
            
                loader = PyPDFLoader(file_name)
            
                documents = loader.load_and_split()
        
            elif '.docx' in doc.name:
                loader = Docx2txtLoader(file_name)
                documents = loader.load_and_split()
            elif '.pptx' in doc.name:
                loader = UnstructuredPowerPointLoader(file_name)
                documents = loader.load_and_split()

            doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, #900
        chunk_overlap=200, #100
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


#def get_conversation_chain(vetorestore,openai_api_key):
#    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
#    conversation_chain = ConversationalRetrievalChain.from_llm(
#            llm=llm, 
#            chain_type="stuff", 
#            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
#            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
#            get_chat_history=lambda h: h,
#            return_source_documents=True,
#            verbose = True
#        )

#    return conversation_chain

'''
def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vetorestore.as_retriever(search_type="mmr"),
        return_source_documents=True
    )
    return qa_chain
'''

def get_conversation_chain(vetorestore):
    # HuggingFaceHub LLM 불러오기
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # 성능 좋은 무료 모델
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type="mmr", vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def safe_query(chain, query):
    for attempt in range(3):  # 최대 3회 시도
        try:
            return chain({"query": query})
        except openai.RateLimitError:
            wait_time = 5 * (attempt + 1)
            st.warning(f"Rate limit 발생 😢 {wait_time}초 대기 후 재시도합니다...")
            time.sleep(wait_time)
    raise Exception("Rate limit 계속 발생 - 잠시 후 다시 시도하세요.")


# =========================
# 안전한 질의 함수 (Rate limit 등 대비)
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


if __name__ == '__main__':
    main()
