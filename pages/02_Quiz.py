import os
import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableLambda


class Quiz:
    def __init__(self, question, index):
        self.question = question["question"]
        self.answers = question["answers"]
        self.select = None
        self.index = index

    def render(self):
        st.write(self.question)
        self.select = st.radio(
            "Select an option",
            [answer["answer"] for answer in self.answers],
            index=None,
            key=self.index,
        )
        if {"answer": self.select, "correct": True} in self.answers:
            st.success("Correct")
            return True
        elif self.select is not None:
            st.error("Wrong")
        return False


def create_quiz(questions):
    all_correct = True
    for i, question in enumerate(questions):
        all_correct = Quiz(question, i).render() and all_correct
    return all_correct


st.set_page_config(page_title="QuizGPT", page_icon="❓")
st.title("QuizGPT")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, _llm, difficulty):
    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that is role playing as a teacher.

                Based ONLY on the following context make 10 {difficulty} difficulty questions to test the user's knowledge about the text.

                Each question should have 4 answers, three of them must be incorrect and one should be correct.

                Context: {context}
                """,
            )
        ]
    )
    chain = (
        {
            "context": format_docs,
            "difficulty": RunnableLambda(lambda _: difficulty.upper()),
        }
        | questions_prompt
        | _llm
    )
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(term)


def format_docs(docs):
    return "\n\n".join([document.page_content for document in docs])


with st.sidebar:
    st.link_button(
        label="https://github.com/ryugibo/nomadcoders.gpt-challenge/blob/main/pages/01_Quiz.py",
        url="https://github.com/ryugibo/nomadcoders.gpt-challenge/blob/main/pages/01_Quiz.py",
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

    difficulty = st.radio("Select difficulty", ["Easy", "Hard"])

    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "upload a .docs, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia..")
        if topic:
            docs = wiki_search(topic)

if open_api_base and open_api_key and docs:
    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

    llm = ChatOpenAI(
        temperature=1e-1,
        model_name="gpt-4o-mini",
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        base_url=open_api_base,
        api_key=open_api_key,
    ).bind(
        function_call={"name": "create_quiz"},
        functions=[function],
    )

    response = run_quiz_chain(docs, llm, difficulty)
    with st.form("questions_form"):
        questions = json.loads(response.additional_kwargs["function_call"]["arguments"])
        all_correct = create_quiz(questions["questions"])
        if all_correct:
            st.balloons()
        st.form_submit_button(disabled=all_correct)
else:
    st.markdown(
        """
        Welcom to Quiz GPT.

        I will make a quiz from WIkipedia articles or files you upload to test
        your knowledge and help you study.

        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
