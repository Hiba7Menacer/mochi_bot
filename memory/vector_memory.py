"""ChromaDB vector memory."""
import chromadb
from config.settings import CHROMA_PATH

class VectorMemory:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        except Exception as e:
            print(f"[VectorMemory] Init error: {e}")
            self.client = None

    def add(self, user_id, text, metadata=None):
        if not self.client:
            return
        try:
            collection = self.client.get_or_create_collection(f"user_{user_id}")
            collection.add(documents=[text], metadatas=[metadata or {}], ids=[f"msg_{hash(text)}"])
        except Exception as e:
            print(f"[VectorMemory] Add error: {e}")

    def search(self, user_id, query, n=3):
        if not self.client:
            return None
        try:
            collection = self.client.get_or_create_collection(f"user_{user_id}")
            return collection.query(query_texts=[query], n_results=n)
        except Exception as e:
            print(f"[VectorMemory] Search error: {e}")
            return None
