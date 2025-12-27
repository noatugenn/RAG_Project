#!/usr/bin/env python3
"""
index_documents.py - Main Document Indexing Entry Point

This is the main script that orchestrates the document indexing process.
It imports and uses all the modular components to:
1. Extract text from PDF/DOCX documents
2. Split text into chunks using configurable strategies
3. Generate vector embeddings using Google Gemini API
4. Store chunks and embeddings in PostgreSQL database

Usage:
    python index_documents.py --file document.pdf --strategy fixed
    python index_documents.py --file document.docx --strategy sentence
    python index_documents.py --file manual.pdf --strategy paragraph -v

Chunking Strategies:
    - fixed: Fixed-size chunks with overlap (default)
    - sentence: Sentence-based splitting
    - paragraph: Paragraph-based splitting

Requirements:
    - PostgreSQL database with document_chunks table
    - .env file with database credentials and Gemini API key
    - All packages from requirements.txt installed
"""

import os
import sys
import argparse
import logging

# Import our modular components
from config import Config
from document_extractor import DocumentExtractor
from text_chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from vector_database import VectorDatabase


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return logging.getLogger(__name__)


def process_document(file_path: str, strategy: str, config: Config,
                     logger: logging.Logger) -> dict:
    """
    Main document processing pipeline.

    This function orchestrates the entire indexing process:
    1. Extract text from document (PDF or DOCX)
    2. Chunk text using specified strategy
    3. Generate embeddings for each chunk
    4. Save to database

    Args:
        file_path: Path to the document file
        strategy: Chunking strategy ('fixed', 'sentence', 'paragraph')
        config: Configuration object with credentials
        logger: Logger instance

    Returns:
        Dictionary with processing statistics:
        - filename: Name of processed file
        - strategy: Chunking strategy used
        - text_length: Total characters extracted
        - chunk_count: Number of chunks created
        - chunks_saved: Number of chunks saved to database
        - success: Whether processing completed successfully
    """
    stats = {
        'filename': os.path.basename(file_path),
        'strategy': strategy,
        'text_length': 0,
        'chunk_count': 0,
        'chunks_saved': 0,
        'success': False
    }

    try:
        # Step 1: Extract text from document
        logger.info(f"Step 1: Extracting text from {file_path}")
        extractor = DocumentExtractor()
        text = extractor.extract(file_path)
        stats['text_length'] = len(text)
        logger.info(f"  Extracted {len(text):,} characters")

        if not text.strip():
            logger.warning("No text extracted from document")
            return stats

        # Step 2: Chunk the text
        logger.info(f"Step 2: Chunking text using '{strategy}' strategy")
        chunker = TextChunker()
        chunks = chunker.chunk(text, strategy)
        stats['chunk_count'] = len(chunks)
        logger.info(f"  Created {len(chunks)} chunks")

        if not chunks:
            logger.warning("No chunks created")
            return stats

        # Get chunk statistics
        chunk_stats = chunker.get_chunk_stats(chunks)
        logger.debug(f"  Chunk stats: avg={chunk_stats['avg_chars']} chars, "
                    f"min={chunk_stats['min_chars']}, max={chunk_stats['max_chars']}")

        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings via Gemini API")
        embedding_gen = EmbeddingGenerator(config.gemini_api_key)
        embeddings = embedding_gen.generate_batch(chunks)
        logger.info(f"  Generated {len(embeddings)} embeddings")

        # Step 4: Save to database
        logger.info("Step 4: Saving to PostgreSQL database")
        filename = os.path.basename(file_path)

        with VectorDatabase(config) as db:
            # Prepare batch data: (chunk_text, embedding, filename, strategy)
            chunks_data = [
                (chunk, emb, filename, strategy)
                for chunk, emb in zip(chunks, embeddings)
            ]

            saved_count = db.save_chunks_batch(chunks_data)
            stats['chunks_saved'] = saved_count

            # Get total count for this file
            total_count = db.get_chunk_count(filename)
            logger.info(f"  Saved {saved_count} chunks. Total for file: {total_count}")

        stats['success'] = True
        logger.info("Processing completed successfully!")
        return stats

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Index documents into vector database for RAG systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python index_documents.py --file document.pdf --strategy fixed
    python index_documents.py --file report.docx --strategy sentence
    python index_documents.py --file manual.pdf --strategy paragraph -v

Strategies:
    fixed     - Split into fixed-size chunks with overlap (default)
    sentence  - Split by sentence boundaries
    paragraph - Split by paragraph boundaries
        """
    )

    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to the document file (PDF or DOCX)'
    )

    parser.add_argument(
        '--strategy', '-s',
        choices=['fixed', 'sentence', 'paragraph'],
        default='fixed',
        help='Chunking strategy (default: fixed)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (debug level)'
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the document indexing script.

    This function:
    1. Parses command-line arguments
    2. Sets up logging
    3. Loads and validates configuration
    4. Processes the document
    5. Prints results summary
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Print header
    print()
    print("=" * 60)
    print("  Document Vector Indexing - Jeen Project")
    print("=" * 60)
    print()

    # Load and validate configuration
    logger.info("Loading configuration from .env file")
    config = Config()

    if not config.validate():
        print()
        print("ERROR: Missing required configuration.")
        print("Please ensure .env file exists with all required variables.")
        print("See .env.example for reference.")
        print()
        sys.exit(1)

    logger.info("Configuration validated successfully")

    # Check if file exists
    if not os.path.exists(args.file):
        print(f"\nERROR: File not found: {args.file}")
        sys.exit(1)

    # Process the document
    try:
        stats = process_document(args.file, args.strategy, config, logger)

        # Print results summary
        print()
        print("=" * 60)
        print("  Processing Complete")
        print("=" * 60)
        print(f"  File:           {stats['filename']}")
        print(f"  Strategy:       {stats['strategy']}")
        print(f"  Text length:    {stats['text_length']:,} characters")
        print(f"  Chunks created: {stats['chunk_count']}")
        print(f"  Chunks saved:   {stats['chunks_saved']}")
        print(f"  Status:         {'SUCCESS' if stats['success'] else 'FAILED'}")
        print("=" * 60)
        print()

        sys.exit(0 if stats['success'] else 1)

    except FileNotFoundError:
        print(f"\nERROR: File not found: {args.file}")
        sys.exit(1)

    except ValueError as e:
        print(f"\nERROR: Invalid input - {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            logger.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
