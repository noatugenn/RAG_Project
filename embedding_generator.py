"""
embedding_generator.py - Gemini API Embedding Generation Module

This module handles generating vector embeddings using Google Gemini API.
It converts text chunks into 768-dimensional vectors for semantic search.

Usage:
    from embedding_generator import EmbeddingGenerator

    generator = EmbeddingGenerator(api_key="your_key")
    embedding = generator.generate_single("Hello world")
    embeddings = generator.generate_batch(["chunk1", "chunk2"])
"""

import time
import logging
from typing import List

import google.generativeai as genai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingGenerator:
    """
    Generates vector embeddings using Google Gemini API.

    This class is responsible for:
    - Configuring the Gemini API client
    - Generating embeddings for single text chunks
    - Batch processing multiple chunks with rate limiting
    - Handling API errors with retry logic

    The embeddings are 768-dimensional vectors that represent
    the semantic meaning of the text.

    Attributes:
        api_key: Google Gemini API key
        model: Embedding model name (default: models/embedding-001)
    """

    # Default embedding model (text-embedding-004 has better free tier limits)
    DEFAULT_MODEL = "models/text-embedding-004"

    # Embedding dimension (text-embedding-004 produces 768-dim vectors)
    EMBEDDING_DIMENSION = 768

    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the embedding generator with API credentials.

        Args:
            api_key: Google Gemini API key
            model: Optional model name (default: models/embedding-001)
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)

        # Configure the Gemini API client
        genai.configure(api_key=api_key)
        self.logger.debug(f"Configured Gemini API with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.

        Uses tenacity for automatic retry with exponential backoff
        in case of temporary API failures.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector (768 dimensions)

        Raises:
            Exception: If API call fails after 3 retries
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided, returning zero vector")
            return [0.0] * self.EMBEDDING_DIMENSION

        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )

            embedding = result['embedding']
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise

    def generate_batch(self, chunks: List[str], delay: float = 0.1,
                       show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple chunks with rate limiting.

        Processes chunks sequentially with a configurable delay between
        API calls to avoid rate limiting. Shows progress bar by default.

        Args:
            chunks: List of text chunks to embed
            delay: Delay between API calls in seconds (default: 0.1)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of embedding vectors (one per chunk)
        """
        if not chunks:
            self.logger.warning("Empty chunk list provided")
            return []

        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = []

        # Create iterator with optional progress bar
        iterator = chunks
        if show_progress:
            iterator = tqdm(chunks, desc="Generating embeddings")

        for chunk in iterator:
            try:
                embedding = self.generate_single(chunk)
                embeddings.append(embedding)

                # Rate limiting delay
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                self.logger.error(f"Failed to generate embedding for chunk: {e}")
                # Add zero vector as placeholder for failed chunks
                embeddings.append([0.0] * self.EMBEDDING_DIMENSION)

        self.logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def test_connection(self) -> bool:
        """
        Test the API connection by generating a simple embedding.

        Returns:
            True if API is working, False otherwise
        """
        try:
            result = self.generate_single("Test connection")
            return len(result) == self.EMBEDDING_DIMENSION
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model name and dimension
        """
        return {
            'model': self.model,
            'dimension': self.EMBEDDING_DIMENSION,
            'task_type': 'retrieval_document'
        }
