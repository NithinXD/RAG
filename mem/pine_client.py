import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Pinecone with connection pooling and retry logic
class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX")
        self.dimension = 768  # Gemini's embedding dimension
        self.pc = None
        self.index = None
        self.max_retries = 3
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.connect()

    def connect(self):
        """Establish connection to Pinecone with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.pc = Pinecone(api_key=self.api_key)

                # Create index if it doesn't exist
                if self.index_name not in self.pc.list_indexes().names():
                    logger.info(f"Creating new Pinecone index: {self.index_name}")
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    # Wait for index to be ready
                    time.sleep(10)

                self.index = self.pc.Index(self.index_name)
                logger.info("Successfully connected to Pinecone")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.critical("Failed to connect to Pinecone after multiple attempts")
                    raise

    def upsert(self, vectors, batch_size=100):
        """Upsert vectors with batching and retry logic"""
        if not vectors:
            return

        # Split into batches
        batches = [vectors[i:i+batch_size] for i in range(0, len(vectors), batch_size)]

        for batch in batches:
            for attempt in range(self.max_retries):
                try:
                    self.index.upsert(vectors=batch)
                    break
                except Exception as e:
                    logger.error(f"Upsert attempt {attempt+1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        # Try reconnecting
                        self.connect()
                    else:
                        logger.error(f"Failed to upsert batch after {self.max_retries} attempts")
                        raise

    @lru_cache(maxsize=100)
    def query(self, vector, top_k=10, include_metadata=True, filter=None, namespace=""):
        """Query the index with caching for repeated queries"""
        for attempt in range(self.max_retries):
            try:
                return self.index.query(
                    vector=vector,
                    top_k=top_k,
                    include_metadata=include_metadata,
                    filter=filter,
                    namespace=namespace
                )
            except Exception as e:
                logger.error(f"Query attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    # Try reconnecting
                    self.connect()
                else:
                    logger.error(f"Failed to query after {self.max_retries} attempts")
                    raise

    def delete(self, ids=None, filter=None, namespace="", delete_all=False):
        """Delete vectors from the index"""
        for attempt in range(self.max_retries):
            try:
                if delete_all:
                    self.index.delete(delete_all=True, namespace=namespace)
                elif filter:
                    self.index.delete(filter=filter, namespace=namespace)
                elif ids:
                    self.index.delete(ids=ids, namespace=namespace)
                return
            except Exception as e:
                logger.error(f"Delete attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    self.connect()
                else:
                    logger.error(f"Failed to delete after {self.max_retries} attempts")
                    raise

# Create a singleton instance
client = PineconeClient()
index = client.index