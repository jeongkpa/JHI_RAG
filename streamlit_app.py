import tiktoken
from loguru import logger
import os
import pandas as pd
from langchain.schema import Document
import faiss
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings.cache import CacheBackedEmbeddings

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# 환경변수 설정
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 미리 정의된 CSV 파일 경로
PRELOADED_CSV = "test.csv"

# 필수 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    documents = []
    for index, row in df.iterrows():
        content = ' '.join(row.values.astype(str))
        document = Document(page_content=content, metadata={'source': file_path, 'row': index})
        documents.append(document)

    return documents

def get_text(file_paths):
    doc_list = []
    for file_path in file_paths:
        doc_list.extend(load_csv(file_path))
    return doc_list

@st.cache_resource()
def embed_documents(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(_docs)

    cache_dir = LocalFileStore("./.cache/embeddings")

    # BGE Embedding: @Mineru
    model_name = "BAAI/bge-m3"
    model_kwargs = {
        "device": "cpu"  # 필요에 따라 "cuda" 또는 "mps"로 변경 가능
    }
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True}
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(split_docs, embedding=cached_embeddings)
    return vectorstore  # vectorstore를 반환

def get_vectorstore(documents):
    return embed_documents(documents)

def get_conversation_chain(vectorstore, openai_api_key, model_selection):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),  # 여기서 as_retriever 호출
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def get_vectorstore(documents):
    return embed_documents(documents)

def get_conversation_chain(vectorstore, openai_api_key, model_selection):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="기업문화 A to Z")
    st.title("무엇이든 물어보세요 A to Z")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"),
            key="model_selection"
        )

    # 애플리케이션 시작 시 임베딩 프로세스 실행
    if st.session_state.processComplete is None:
        with st.spinner("데이터를 처리하고 있습니다. 이 작업은 수 분이 소요되니 커피 한잔 하고 오세요. ☕️"):
            documents = get_text([PRELOADED_CSV])
            vectorstore = get_vectorstore(documents)
            st.session_state.conversation = get_conversation_chain(vectorstore, OPENAI_API_KEY, st.session_state.model_selection)
            st.session_state.processComplete = True
        st.success("데이터 처리가 완료되었습니다!")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요, 기업문화 ChatBot Beta입니다. 😊"
                       "궁금한 것을 질문해 주세요. ❓"
                       "  \n아직은 걸음마 단계이니 잘 부탁드려요. 👣"
                       "저는 앞으로 더 성장할 거에요. 🌱"
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("회사제도, 복리후생, 근무환경 등 궁금한 것을 질문하세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("관련된 문서를 찾는 중입니다..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("기존 유사 Reference"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()