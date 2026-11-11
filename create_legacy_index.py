import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc.create_index(
    name=os.environ["INDEX_NAME"],
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("Standard vector index created successfully!")
print(pc.describe_index(os.environ["INDEX_NAME"]))


#===========================================================#
#  EXECUTE SCRIPT TO CREATE [STANDARD/LEGACY] VECTOR INDEX
#===========================================================#

# SCRIPT: uv run create_legacy_index.py
