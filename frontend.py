import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from create_llm import load_llm,set_custom_template,CUSTOM_PROMPT_TEMPLATE

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db



def main():
    st.title("Heath Bot  Inference API Test")
    prompt=st.chat_input("Ask me anything related to the health:")

    # use sesstion state to store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    if prompt:
        st.chat_message("user").markdown(prompt)
        # Append the user message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})


        llm=load_llm()

        try:
            vector_store=load_vectorstore()
            if vector_store is None:
                st.error("Failed to load vector store.")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                chain_type_kwargs={"prompt": set_custom_template(CUSTOM_PROMPT_TEMPLATE)},
                return_source_documents=True,
            )
            result = qa_chain({"query": prompt})
            responce=result['result']

        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return


        # responce="This is a test response."
        st.chat_message("assistant").markdown(responce)
        # Append the assistant message to the session state
        st.session_state.messages.append({"role": "assistant", "content": responce})

if __name__ == "__main__":
    main()
