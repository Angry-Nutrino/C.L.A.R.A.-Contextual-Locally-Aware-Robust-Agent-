import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Loading the pdf

path= "./docs"
DB_PATH="./knowledge_base"

def build_knowledge_base()->str:
    loader= PyPDFDirectoryLoader(path, glob="**/*.pdf")
    raw_docs=loader.load()

    if not raw_docs:
        print("❌ No PDFs found!")
        return
    
    
    splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks= splitter.split_documents(raw_docs)

    embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(chunks,embeddings)

    vectorstore.save_local(DB_PATH)
    print(f"Model saved at DB PATH: {DB_PATH}")

if __name__ == "__main__":
    build_knowledge_base()


    