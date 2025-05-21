import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Any
from huggingface_hub import InferenceClient
from fastapi import FastAPI


# Validate HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Set it with: $env:HF_TOKEN='your_huggingface_api_token'")

hugging_face_repo = "google/gemma-2b-it"

# Custom LLM class to use InferenceClient.chat_completion
# Fix 1: Added client as an Optional field with default None to satisfy Pydantic
# Why: Pydantic requires all fields to be declared and initialized for validation
class HuggingFaceChat(LLM):
    model: str
    token: str
    client: Optional[InferenceClient] = None  # Declare client with default None

    def __init__(self, model: str, token: str):
        super().__init__(model=model, token=token)
        self.client = InferenceClient(model=model, token=token)  # Initialize client after super()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error in chat_completion: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"

def load_llm(hugging_face_repo_id=hugging_face_repo):
    try:
        # Fix 2: Use corrected HuggingFaceChat class
        llm = HuggingFaceChat(model=hugging_face_repo_id, token=HF_TOKEN)
        return llm
    except Exception as e:
        raise ValueError(f"Failed to initialize HuggingFaceChat: {str(e)}")

# Define FAISS path and prompt template
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

# Load embeddings and FAISS database
# Fix 3: Kept robust error handling for FAISS loading
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    raise FileNotFoundError(f"Failed to load FAISS database from {DB_FAISS_PATH}: {str(e)}")

# Create QA chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(hugging_face_repo),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": set_custom_template(CUSTOM_PROMPT_TEMPLATE)},
        return_source_documents=True,
    )
except Exception as e:
    raise ValueError(f"Failed to create RetrievalQA chain: {str(e)}")

# Test the LLM
try:
    llm = load_llm(hugging_face_repo)
    response = qa_chain.invoke("Who is the best football player?")
    print("Response:", response["result"])
    print("Source Documents:", response["source_documents"])
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc()
