import os
import pandas as pd

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_pinecone import PineconeVectorStore # type: ignore

def generateEmbeddings(df: pd.DataFrame, INDEX_NAME):

    # Embedding model - matches the index dimension (384)
    embeddingModel = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Connect to Pinecone
    pc = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])

    # 1️⃣ Check if index already exists
    if INDEX_NAME in pc.list_indexes().names():
        print("🔄 Pinecone Index Exists")
        vectorStore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddingModel
        )
        print("✅ Loaded Existing Pinecone Vector Store...")
        return vectorStore

    # 2️⃣ If index does NOT exist → create and upload embeddings
    print("🆕 Index Not Found — Creating New Pinecone Index...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,          # MiniLM-L6-v2 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    index = pc.Index(INDEX_NAME)

    print("⏳ Preparing Documents...")
    documents = []

    for _, row in df.iterrows():
        for i in range(1, 6):
            text = row.get(f"translation_{i}")
            if isinstance(text, str) and text.strip():
                metadata = {
                    "verse_no": row["verse_no"],
                    "spoken_by": row["spoken_by"],
                    "sanskrit_text": row["sanskrit_text"],
                    "translation_index": i
                }
                documents.append(Document(page_content=text, metadata=metadata))

    print("⏳ Uploading Embeddings To Pinecone...")

    vectorStore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddingModel,
        index_name=INDEX_NAME
    )

    print("✅ New Pinecone Vector Store Created & Ready.")
    return vectorStore