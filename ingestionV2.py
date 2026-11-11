
import os
import hashlib

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":

    # Step 1: Load documents
    print("Ingesting...")

    loader = UnstructuredLoader(
        file_path="/Users/kinzabukhari/Documents/Abdul Learning/langchain-course/mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    # Step 2: Split documents
    print("Splitting...")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    # Step 3: Connect to Pinecone
    print("Connecting to Pinecone...")

    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )

    index = pc.index(os.environ["INDEX_NAME"])

    # Step 4: Prepare records
    print("Preparing records...")

    records = []

    for i, doc in enumerate(texts):

        content = doc.page_content
        source = doc.metadata.get("source", "mediumblog1.txt")

        record_id = hashlib.sha256(
            f"{source}-{i}-{content}".encode("utf-8")
        ).hexdigest()

        records.append({
            "_id": record_id,
            "title": os.path.basename(source),
            "chunk_text": content
        })

    # Step 5: Upload documents
    print("Uploading documents to Pinecone...")

    response = index.documents.upsert(
        namespace="medium-blogs",
        documents=records
    )

    print(f"Successfully uploaded {response.upserted_count} documents")

    print("Ingestion completed!")
