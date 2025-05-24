import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from create_llm import load_llm, set_custom_template, CUSTOM_PROMPT_TEMPLATE

# Path to the local FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the vector store with embeddings
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Main Streamlit app function
def main():
    st.title("🩺 Health Bot - Inference API Test")

    # Chat input field
    prompt = st.chat_input("Ask me anything related to health:")

    # Initialize chat history using session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Process new user prompt
    if prompt:
        # Display user message and add to session
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Load LLM and vector store
        llm = load_llm()

        try:
            vector_store = load_vectorstore()
            if not vector_store:
                st.error("⚠️ Failed to load vector store.")
                return

            # Create a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                chain_type_kwargs={"prompt": set_custom_template(CUSTOM_PROMPT_TEMPLATE)},
                return_source_documents=True,
            )

            # Run the chain with the user's query
            result = qa_chain({"query": prompt})
            response = result["result"]

        except Exception as e:
            st.error(f"🚨 Error: {e}")
            return

        # Display the assistant's response and add to session
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Entry point
if __name__ == "__main__":
    main()
