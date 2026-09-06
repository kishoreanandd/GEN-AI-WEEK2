import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the .env file.")
    st.stop()


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# --------------------------------------------------
# Initialize models
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# --------------------------------------------------
# PDF uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Process uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    # Check whether this is a new PDF
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("file_id") != file_id:

        with st.spinner(
            "Reading PDF, creating chunks and building vector database..."
        ):

            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_pdf_path = temp_file.name

            # ------------------------------------------
            # Load PDF
            # ------------------------------------------

            loader = PyPDFLoader(temp_pdf_path)

            documents = loader.load()

            # ------------------------------------------
            # Split PDF into chunks
            # ------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(documents)

            # ------------------------------------------
            # Create embeddings and FAISS database
            # ------------------------------------------

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            # ------------------------------------------
            # Save FAISS database locally
            # ------------------------------------------

            db_path = "faiss_index"

            vectorstore.save_local(
                db_path
            )

            # ------------------------------------------
            # Create retriever
            # ------------------------------------------

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 4
                }
            )

            # ------------------------------------------
            # Create prompt
            # ------------------------------------------

            prompt = ChatPromptTemplate.from_template(
                """
                Answer the user's question using only the
                information contained in the provided context.

                If the answer cannot be found in the context,
                clearly say:

                "I couldn't find the answer in the uploaded PDF."

                Do not make up information.

                Context:
                {context}

                Question:
                {input}

                Answer:
                """
            )

            # ------------------------------------------
            # Create document chain
            # ------------------------------------------

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            # ------------------------------------------
            # Create retrieval chain
            # ------------------------------------------

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            # Store everything in session state
            st.session_state.file_id = file_id
            st.session_state.vectorstore = vectorstore
            st.session_state.retrieval_chain = retrieval_chain
            st.session_state.chunks = chunks

        st.success(
            f"PDF processed successfully. "
            f"Created {len(chunks)} chunks."
        )

    else:

        st.success(
            "PDF is already processed and ready for questions."
        )


# --------------------------------------------------
# Question input
# --------------------------------------------------

if "retrieval_chain" in st.session_state:

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is this document about?"
    )

    if question:

        with st.spinner("Searching the PDF and generating answer..."):

            response = st.session_state.retrieval_chain.invoke(
                {
                    "input": question
                }
            )

        # ------------------------------------------
        # Display answer
        # ------------------------------------------

        st.subheader("Answer")

        st.write(
            response["answer"]
        )

        # ------------------------------------------
        # Display retrieved sources
        # ------------------------------------------

        with st.expander("View retrieved PDF pages"):

            documents = response.get(
                "context",
                []
            )

            for i, document in enumerate(documents):

                page_number = (
                    document.metadata.get("page", "Unknown")
                )

                st.markdown(
                    f"**Source {i + 1} | Page {page_number + 1 if isinstance(page_number, int) else page_number}**"
                )

                st.write(
                    document.page_content[:1000]
                )

                st.divider()