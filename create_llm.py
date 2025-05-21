from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")
# hugging_face/_repo = "tiiuae/falcon-7b-instruct"  # Or any other text-generation model
hugging_face_repo = "google/flan-t5-small"


def load_llm(hugging_face_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=hugging_face_repo_id,
        temperature=0.5,
         huggingfacehub_api_token=HF_TOKEN,
         task="text2text-generation"
    )
      # Set the token directly on the client if applicable
    return llm

# Step 2: Create Prompt and FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of the information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_template(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(hugging_face_repo),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": set_custom_template(CUSTOM_PROMPT_TEMPLATE)},
    return_source_documents=True,
)

# Now invoke with 

import traceback

# USER_QUESTION = input("Enter your question: ")
# try:
#     result = qa_chain.invoke({"query": USER_QUESTION})
#     print(result)
# except Exception as e:
#     print("Error:", str(e))
#     traceback.print_exc()
llm = load_llm(hugging_face_repo)
response = llm.invoke("What is AI?")
print(response)



 # Use invoke instead of __call__
