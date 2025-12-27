"""
text_chunker.py - Text Splitting Strategies Module

This module handles splitting text into chunks using different strategies.
It provides three chunking strategies for RAG systems:
1. Fixed-size with overlap
2. Sentence-based splitting
3. Paragraph-based splitting

Usage:
    from text_chunker import TextChunker

    chunker = TextChunker()
    chunks = chunker.chunk(text, strategy='fixed')
    # or
    chunks = chunker.chunk_by_sentences(text)
"""

import logging
from typing import List

import nltk


class TextChunker:
    """
    Splits text into chunks using different strategies.

    This class is responsible for:
    - Fixed-size chunking with configurable overlap
    - Sentence-based chunking using NLTK tokenizer
    - Paragraph-based chunking using newline detection

    Each strategy has different use cases:
    - fixed: Good for generic documents, consistent chunk sizes
    - sentence: Good for well-formatted text, preserves complete thoughts
    - paragraph: Good for structured documents, preserves logical sections
    """

    # Available chunking strategies
    STRATEGIES = ['fixed', 'sentence', 'paragraph']

    def __init__(self):
        """Initialize the text chunker and ensure NLTK data is available."""
        self.logger = logging.getLogger(__name__)

        # Ensure NLTK punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

    def chunk_fixed_size(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.

        Creates chunks of approximately equal size with overlapping characters
        between consecutive chunks to maintain context.

        Example:
            Text: "Hello world, this is a test"
            chunk_size=10, overlap=3

            Chunk 1: "Hello worl"  (chars 0-10)
            Chunk 2: "rld, this "  (chars 7-17, "rld" overlaps)

        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk in characters (default: 500)
            overlap: Number of overlapping characters between chunks (default: 50)

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        self.logger.debug(f"Fixed-size chunking: size={chunk_size}, overlap={overlap}")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get chunk of specified size
            end = start + chunk_size
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())

            # Move start position, accounting for overlap
            start = end - overlap

            # Avoid infinite loop if overlap >= chunk_size
            if overlap >= chunk_size:
                start = end

        self.logger.debug(f"Created {len(chunks)} fixed-size chunks")
        return chunks

    def chunk_by_sentences(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text by sentence boundaries, grouping sentences until max size.

        Uses NLTK sentence tokenizer to identify sentence boundaries.
        Groups consecutive sentences together until adding another sentence
        would exceed the maximum chunk size.

        Example:
            Text: "Hello. This is a test. Chunking works great."
            max_chunk_size=30

            Chunk 1: "Hello. This is a test."
            Chunk 2: "Chunking works great."

        Args:
            text: Input text to chunk
            max_chunk_size: Maximum size of each chunk in characters (default: 500)

        Returns:
            List of text chunks (each containing complete sentences)
        """
        if not text or not text.strip():
            return []

        self.logger.debug(f"Sentence-based chunking: max_size={max_chunk_size}")

        # Use NLTK sentence tokenizer
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            self.logger.warning(f"NLTK tokenization failed, falling back to simple split: {e}")
            # Fallback: split on period followed by space
            sentences = [s.strip() + '.' for s in text.split('. ') if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            # If single sentence exceeds max size, add it as its own chunk
            if sentence_size > max_chunk_size:
                # First, save current chunk if it exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # Add the long sentence as its own chunk
                chunks.append(sentence)
                continue

            # If adding sentence would exceed max size, start new chunk
            if current_size + sentence_size + 1 > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        self.logger.debug(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text by paragraph boundaries (double newlines).

        Identifies paragraphs by looking for double newline characters.
        Each paragraph becomes its own chunk.

        Example:
            Text: "First paragraph here.\\n\\nSecond paragraph here."

            Chunk 1: "First paragraph here."
            Chunk 2: "Second paragraph here."

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks (each is a paragraph)
        """
        if not text or not text.strip():
            return []

        self.logger.debug("Paragraph-based chunking")

        # Split on double newlines (paragraph boundaries)
        paragraphs = text.split('\n\n')

        # Clean and filter empty paragraphs
        chunks = []
        for para in paragraphs:
            # Clean up whitespace within paragraph
            cleaned = ' '.join(para.split())
            if cleaned:
                chunks.append(cleaned)

        self.logger.debug(f"Created {len(chunks)} paragraph-based chunks")
        return chunks

    def chunk(self, text: str, strategy: str = 'fixed', **kwargs) -> List[str]:
        """
        Main chunking method that calls the appropriate strategy.

        This is the recommended method to use. It routes to the correct
        chunking strategy based on the strategy parameter.

        Args:
            text: Input text to chunk
            strategy: Chunking strategy - 'fixed', 'sentence', or 'paragraph'
            **kwargs: Additional arguments passed to the strategy function
                     - For 'fixed': chunk_size, overlap
                     - For 'sentence': max_chunk_size
                     - For 'paragraph': (no additional args)

        Returns:
            List of text chunks

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Choose from: {self.STRATEGIES}"
            )

        self.logger.info(f"Chunking text using '{strategy}' strategy")

        if strategy == 'fixed':
            return self.chunk_fixed_size(text, **kwargs)
        elif strategy == 'sentence':
            return self.chunk_by_sentences(text, **kwargs)
        elif strategy == 'paragraph':
            return self.chunk_by_paragraphs(text)

    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Get statistics about a list of chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with count, total_chars, avg_chars, min_chars, max_chars
        """
        if not chunks:
            return {
                'count': 0,
                'total_chars': 0,
                'avg_chars': 0,
                'min_chars': 0,
                'max_chars': 0
            }

        lengths = [len(chunk) for chunk in chunks]

        return {
            'count': len(chunks),
            'total_chars': sum(lengths),
            'avg_chars': sum(lengths) // len(chunks),
            'min_chars': min(lengths),
            'max_chars': max(lengths)
        }
