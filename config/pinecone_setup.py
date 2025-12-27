import os
from pinecone import Pinecone, ServerlessSpec
import json
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeSetup:
    """
    Pinecone database initialization and management for medical chatbot
    """

    def __init__(self, api_key: str = None, environment: str = "us-east-1-aws"):
        """
        Initialize Pinecone connection

        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
        """
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment
        self.pc = None

        if not self.api_key:
            raise ValueError("Pinecone API key is required")

        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Pinecone connection"""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Successfully initialized Pinecone connection")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

    def create_index(self, 
                    index_name: str = "medical-chatbot-index",
                    dimension: int = 1536,
                    metric: str = "cosine",
                    cloud: str = "aws",
                    region: str = "us-east-1") -> bool:
        """
        Create a new Pinecone index

        Args:
            index_name (str): Name of the index
            dimension (int): Vector dimensions (1536 for OpenAI embeddings)
            metric (str): Distance metric
            cloud (str): Cloud provider
            region (str): Cloud region

        Returns:
            bool: True if successful
        """
        try:
            # Check if index already exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if index_name in existing_indexes:
                logger.info(f"Index '{index_name}' already exists")
                return True

            # Create new index
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )

            logger.info(f"Successfully created index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False

    def get_index(self, index_name: str = "medical-chatbot-index"):
        """
        Get Pinecone index instance

        Args:
            index_name (str): Name of the index

        Returns:
            Pinecone Index instance
        """
        try:
            return self.pc.Index(index_name)
        except Exception as e:
            logger.error(f"Failed to get index: {str(e)}")
            raise

    def upsert_vectors(self, 
                      index_name: str,
                      vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors to the index

        Args:
            index_name (str): Name of the index
            vectors (List[Dict]): List of vector dictionaries

        Returns:
            bool: True if successful
        """
        try:
            index = self.get_index(index_name)
            index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            return False

    def query_vectors(self, 
                     index_name: str,
                     vector: List[float],
                     top_k: int = 5,
                     filter_dict: Dict = None,
                     include_metadata: bool = True) -> Dict:
        """
        Query vectors from the index

        Args:
            index_name (str): Name of the index
            vector (List[float]): Query vector
            top_k (int): Number of results to return
            filter_dict (Dict): Filter criteria
            include_metadata (bool): Include metadata in results

        Returns:
            Dict: Query results
        """
        try:
            index = self.get_index(index_name)
            results = index.query(
                vector=vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query vectors: {str(e)}")
            return {}

# Configuration settings
PINECONE_CONFIG = {
    "api_key": os.getenv('PINECONE_API_KEY'),
    "environment": "us-east-1-aws",
    "index_name": "medical-chatbot-index",
    "dimension": 1536,
    "metric": "cosine",
    "cloud": "aws",
    "region": "us-east-1"
}

def initialize_pinecone():
    """Initialize Pinecone database for medical chatbot"""
    try:
        pinecone_setup = PineconeSetup(
            api_key=PINECONE_CONFIG["api_key"],
            environment=PINECONE_CONFIG["environment"]
        )

        # Create index if it doesn't exist
        success = pinecone_setup.create_index(
            index_name=PINECONE_CONFIG["index_name"],
            dimension=PINECONE_CONFIG["dimension"],
            metric=PINECONE_CONFIG["metric"],
            cloud=PINECONE_CONFIG["cloud"],
            region=PINECONE_CONFIG["region"]
        )

        if success:
            logger.info("Pinecone initialization completed successfully")
            return pinecone_setup
        else:
            raise Exception("Failed to initialize Pinecone")

    except Exception as e:
        logger.error(f"Pinecone initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize Pinecone database
    pinecone_db = initialize_pinecone()
    print("Pinecone database initialized successfully!")
