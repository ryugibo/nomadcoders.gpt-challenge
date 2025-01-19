import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

st.title("Fullstack GPT Challenge - RAG")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, base_url, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url=base_url,
        api_key=api_key,
    )

    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedding,
        cache_dir,
    )

    vectorstore = FAISS.from_documents(docs, cache_embeddings)

    return vectorstore.as_retriever()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def invoke_chain(chain, memory, question):
    result = chain.invoke(question)
    memory.save_context({"input": question}, {"output": result.content})
    return result


with st.sidebar:
    st.link_button(
        label="https://github.com/ryugibo/nomadcoders.gpt-challenge/blob/main/pages/01_RAG.py",
        url="https://github.com/ryugibo/nomadcoders.gpt-challenge/blob/main/pages/01_RAG.py",
    )

    open_api_base = st.text_input(
        "OPENAI API BASE URL",
        value=os.getenv(
            "OPENAI_API_BASE",
            "https://api.openai.com/v1",
        ),
    )
    open_api_key = st.text_input(
        "OPENAI API KEY",
        value=os.getenv("OPENAI_API_KEY"),
    )

if open_api_base and open_api_key:
    if "llm" not in st.session_state:
        st.session_state["llm"] = ChatOpenAI(
            model_name="gpt-4o-mini",
            streaming=True,
            callbacks=[ChatCallbackHandler()],
            base_url=open_api_base,
            api_key=open_api_key,
        )
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            return_messages=True,
        )
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

    st.header("Document GPT")
    st.subheader(
        """
        Welcome!

        Use this chatbot to ask questions to an AI about your files!

        Upload your files on the sidebar.
        """
    )

    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file.",
            [
                "pdf",
                "txt",
                "docx",
            ],
        )

    if file:
        retriever = embed_file(
            file,
            open_api_base,
            open_api_key,
        )

        send_message("I'm ready! Ask away!", "ai", save=False)

        paint_history()

        message = st.chat_input("Ask anything about your file...")

        if message:
            send_message(message, "human")

            memory = st.session_state["memory"]
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(
                        lambda _: memory.load_memory_variables({})["history"]
                    ),
                }
                | st.session_state["prompt"]
                | st.session_state["llm"]
            )
            with st.chat_message("ai"):
                response = invoke_chain(chain, st.session_state["memory"], message)
    else:
        st.session_state["messages"] = []

else:
    st.header("Please, check entered url and key in sidebar.")
