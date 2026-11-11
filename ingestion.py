import os
from dotenv import load_dotenv
from langchain_core import embeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":
    
    # Delete all existing vectors in the default namespace
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["INDEX_NAME"])
    index.delete(delete_all=True)
    print("<===All Existing Vectors Deleted===>")

    print("Loding Document...")
    loader = UnstructuredLoader(file_path="/Users/kinzabukhari/Documents/Abdul Learning/langchain-course/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b") # Local model for embedding

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Finish")