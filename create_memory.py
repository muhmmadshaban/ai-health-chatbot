import pandas as pd
import glob
import os
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS

DATA_PATH = r"C:\Users\Muhmmad shaban\Desktop\DATA"

# Document class to hold content, metadata, and an ID
class Document:
    def __init__(self, question, answer, doc_id):
        self.id = doc_id  # Unique identifier for the document
        self.page_content = f"Q: {question} A: {answer}"
        self.metadata = {"question": question, "answer": answer}  # Add metadata

def load_csv_files(data_path):
    # Create a list to hold all Document objects
    all_documents = []
    
    # Use glob to find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    # Loop through the list of CSV files and read them into Document objects
    for file in csv_files:
        # Check if the file is empty
        if os.path.getsize(file) > 0:
            try:
                # Read the CSV file and specify the columns to use
                df = pd.read_csv(file, usecols=["Question", "Answer"])
                for index, row in df.iterrows():
                    # Create a unique ID by combining the file name and index
                    unique_id = f"{os.path.basename(file)}_{index}"
                    all_documents.append(Document(row['Question'], row['Answer'], doc_id=unique_id))  # Assign a unique ID
            except pd.errors.EmptyDataError:
                print(f"Warning: The file {file} is empty and will be skipped.")
            except ValueError as ve:
                print(f"Warning: {ve} in file {file}. Ensure it has 'Question' and 'Answer' columns.")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"Warning: The file {file} is empty and will be skipped.")
    
    return all_documents

# Example usage
documents = load_csv_files(DATA_PATH)
print(f"Loaded {len(documents)} documents.")

# Create chunks of data
def create_chunks(documents, chunk_size=500):
    # Create a list to hold the chunks
    chunks = []
    
    # Loop through the list of Document objects in steps of chunk_size
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

# Example usage
chunks = create_chunks(documents, chunk_size=500)
print(f"Created {len(chunks)} chunks of data.")

# Create embeddings
def get_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    return embedding_model  

embedding_model = get_embeddings()

# Create vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(documents, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"Vector store saved to {DB_FAISS_PATH}.")