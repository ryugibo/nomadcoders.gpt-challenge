import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SitemapLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


def get_answers(inputs, llm):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_prompt = ChatPromptTemplate.from_template(
        """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Make sure to always answer starts with 'Answer:'
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.\nScore: 5

    Question: How far away is the sun?
    Answer: I don't know\nScore: 0

    Question: {question}
"""
    )
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Select the answer that have the highest score. If there are multiple highest scoring answers, choose the most recent one.
            Site sources and return the sources of the answers as they are, do not change them.
            Answers:
            {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs, llm):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    text = soup.find("main").get_text()
    return text


@st.cache_data(show_spinner="Loading websites...")
def load_website(site_url, _base_url, _api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        site_url,
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/workers\/).*",
        ],
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=_base_url,
            api_key=_api_key,
        ),
    )
    return vectorstore.as_retriever()


st.set_page_config(page_title="SiteGPT", page_icon="🌥️")
st.title("SiteGPT")

with st.sidebar:

    st.link_button(
        label="https://github.com/ryugibo/nomadcoders.gpt-challenge/blob/main/pages/03_Site.py",
        url="https://github.com/ryugibo/nomadcoders.gpt-challenge/blob/main/pages/03_Site.py",
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
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=1e-1,
        base_url=open_api_base,
        api_key=open_api_key,
    )
    retriever = load_website(
        "https://developers.cloudflare.com/sitemap.xml",
        open_api_base,
        open_api_key,
    )
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda x: get_answers(x, llm))
            | RunnableLambda(lambda x: choose_answer(x, llm))
        )

        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
else:
    st.warning("Input open ai api key in sidebar")
