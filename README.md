# Jeen Project - Document Vector Indexing

Phase 2 implementation: A modular Python system that processes documents (PDF/DOCX), generates vector embeddings using Google Gemini API, and stores them in PostgreSQL for semantic search.

## 🎯 Project Goal

Build a RAG (Retrieval Augmented Generation) system foundation that:
- Extracts text from PDF and DOCX files
- Splits text into chunks using 3 different strategies
- Generates 768-dimensional embeddings using Google Gemini API
- Stores chunks and embeddings in PostgreSQL database

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run the setup script (creates virtual environment and installs dependencies)
./setup_env.sh

# Activate the virtual environment
source jeen_project/bin/activate
```

### 2. Configure Credentials

Create a `.env` file in the project root:

```bash
nano .env
```

Add the following content (replace with your actual values):

```env
# PostgreSQL Database Configuration
PG_HOST=localhost
PG_PORT=5433
PG_DATABASE=jeen_db
PG_USER=admin
PG_PASSWORD=your_database_password_here

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
```

**How to get your Gemini API key:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

### 3. Verify Setup
```bash
python verify_setup.py
```

### 4. Index Documents
```bash
python index_documents.py --file your_document.pdf --strategy fixed
```

---

## 📖 Usage

### Command Line Interface

```
usage: index_documents.py [-h] --file FILE
                          [--strategy {fixed,sentence,paragraph}] [--verbose]

Index documents into vector database for RAG systems

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to the document file (PDF or DOCX)
  --strategy {fixed,sentence,paragraph}, -s {fixed,sentence,paragraph}
                        Chunking strategy (default: fixed)
  --verbose, -v         Enable verbose logging (debug level)

Examples:
    python index_documents.py --file document.pdf --strategy fixed
    python index_documents.py --file report.docx --strategy sentence
    python index_documents.py --file manual.pdf --strategy paragraph -v

Strategies:
    fixed     - Split into fixed-size chunks with overlap (default)
    sentence  - Split by sentence boundaries
    paragraph - Split by paragraph boundaries
```

### Example Commands

```bash
# Index a PDF with fixed-size chunking (default)
python index_documents.py --file document.pdf

# Index a DOCX with sentence-based chunking
python index_documents.py --file report.docx --strategy sentence

# Index with paragraph-based chunking and verbose logging
python index_documents.py --file manual.pdf --strategy paragraph -v
```

### Example Output

```
============================================================
  Document Vector Indexing - Jeen Project
============================================================

2025-12-26 13:50:56 - INFO - Loading configuration from .env file
2025-12-26 13:50:56 - INFO - Configuration validated successfully
2025-12-26 13:50:56 - INFO - Step 1: Extracting text from solutions.pdf
2025-12-26 13:50:56 - INFO - Extracting text from PDF: solutions.pdf
2025-12-26 13:50:57 - INFO - Total extracted: 2720 characters from 2 pages
2025-12-26 13:50:57 - INFO -   Extracted 2,720 characters
2025-12-26 13:50:57 - INFO - Step 2: Chunking text using 'fixed' strategy
2025-12-26 13:50:57 - INFO - Chunking text using 'fixed' strategy
2025-12-26 13:50:57 - INFO -   Created 7 chunks
2025-12-26 13:50:57 - INFO - Step 3: Generating embeddings via Gemini API
2025-12-26 13:50:57 - INFO - Generating embeddings for 7 chunks
Generating embeddings: 100%|██████████| 7/7 [00:03<00:00,  1.86it/s]
2025-12-26 13:51:00 - INFO - Generated 7 embeddings
2025-12-26 13:51:00 - INFO -   Generated 7 embeddings
2025-12-26 13:51:00 - INFO - Step 4: Saving to PostgreSQL database
2025-12-26 13:51:00 - INFO - Batch saved 7 chunks
2025-12-26 13:51:00 - INFO -   Saved 7 chunks. Total for file: 7
2025-12-26 13:51:00 - INFO - Processing completed successfully!

============================================================
  Processing Complete
============================================================
  File:           solutions.pdf
  Strategy:       fixed
  Text length:    2,720 characters
  Chunks created: 7
  Chunks saved:   7
  Status:         SUCCESS
============================================================
```

### Example Database Data

After indexing, the database contains:

```
 id |   filename    | split_strategy | text_len
----+---------------+----------------+----------
  1 | solutions.pdf | fixed          |      498
  2 | solutions.pdf | fixed          |      498
  3 | solutions.pdf | fixed          |      497
  4 | solutions.pdf | fixed          |      500
  5 | solutions.pdf | fixed          |      500
  6 | solutions.pdf | fixed          |      466
  7 | solutions.pdf | fixed          |       19
(7 rows)
```

---

## 🏗️ Architecture

### Modular Design

The project uses a modular architecture where each component has a single responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                     index_documents.py                       │
│                    (MAIN ENTRY POINT)                        │
│                                                              │
│  python index_documents.py --file doc.pdf --strategy fixed  │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
   │  config.py    │  │ document_     │  │ text_         │
   │               │  │ extractor.py  │  │ chunker.py    │
   │  class Config │  │               │  │               │
   │  - load .env  │  │ class Doc...  │  │ class Text... │
   │  - validate   │  │ - PDF extract │  │ - fixed       │
   └───────────────┘  │ - DOCX extract│  │ - sentence    │
                      └───────────────┘  │ - paragraph   │
                                         └───────────────┘
           │                  │
           ▼                  ▼
   ┌───────────────┐  ┌───────────────┐
   │ embedding_    │  │ vector_       │
   │ generator.py  │  │ database.py   │
   │               │  │               │
   │ class Embed...│  │ class Vector..│
   │ - Gemini API  │  │ - connect     │
   │ - rate limit  │  │ - save chunks │
   └───────────────┘  └───────────────┘
```

### Module Descriptions

| Module | Class | Responsibility |
|--------|-------|----------------|
| `config.py` | `Config` | Load .env file, validate credentials |
| `document_extractor.py` | `DocumentExtractor` | Extract text from PDF/DOCX files |
| `text_chunker.py` | `TextChunker` | Split text using 3 strategies |
| `embedding_generator.py` | `EmbeddingGenerator` | Generate embeddings via Gemini API |
| `vector_database.py` | `VectorDatabase` | PostgreSQL database operations |
| `index_documents.py` | Main | CLI interface, orchestration |

---

## 📁 Project Structure

```
Jeen-Project/
├── index_documents.py      # Main entry point - CLI interface
├── config.py               # Configuration management
├── document_extractor.py   # PDF/DOCX text extraction
├── text_chunker.py         # Text chunking strategies
├── embedding_generator.py  # Gemini API integration
├── vector_database.py      # PostgreSQL operations
├── verify_setup.py         # Setup verification script
├── setup_env.sh            # Automated environment setup
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this - see Quick Start)
├── .gitignore              # Git exclusions
└── README.md               # This file
```

---

## 🔧 Chunking Strategies

### 1. Fixed-Size (`--strategy fixed`)
Splits text into fixed-size chunks with overlap to maintain context.

```
Text: "Hello world, this is a test of chunking"
chunk_size=10, overlap=3

Chunk 1: "Hello worl"     (chars 0-10)
Chunk 2: "rld, this "     (chars 7-17, overlap)
Chunk 3: "is is a te"     (chars 14-24, overlap)
```
**Best for:** Generic documents, consistent chunk sizes

### 2. Sentence-Based (`--strategy sentence`)
Splits at sentence boundaries, grouping sentences until reaching max size.

```
Text: "Hello. This is a test. Chunking works great."

Chunk 1: "Hello. This is a test."
Chunk 2: "Chunking works great."
```
**Best for:** Well-formatted documents with clear sentences

### 3. Paragraph-Based (`--strategy paragraph`)
Splits at paragraph boundaries (double newlines).

```
Text: "First paragraph here.\n\nSecond paragraph here."

Chunk 1: "First paragraph here."
Chunk 2: "Second paragraph here."
```
**Best for:** Documents with logical paragraph structure

---

## 🗄️ Database Schema

```sql
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding JSONB,           -- 768-dimensional vector as JSON array
    filename VARCHAR(500),
    split_strategy VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📊 Progress Log

### 2025-12-26 - BIG MISSION 2: Implement Document Indexing System ✅
**What we did:** Created complete modular document indexing system
**Files added:**
- `config.py` - Configuration management class
- `document_extractor.py` - PDF/DOCX text extraction class
- `text_chunker.py` - 3 chunking strategies class
- `embedding_generator.py` - Gemini API integration class
- `vector_database.py` - PostgreSQL operations class
- `index_documents.py` - Main CLI entry point

**Key features:**
- Modular architecture with single-responsibility classes
- Support for PDF and DOCX document formats
- Three chunking strategies: fixed, sentence, paragraph
- Gemini API embeddings with rate limiting and retry logic
- PostgreSQL storage with JSONB for embeddings
- Progress bars for long operations
- Verbose logging option

**Status:** ✅ Complete

### 2025-12-26 - Mission 1.13: Create Setup Verification Script ✅
**What we did:** Created `verify_setup.py` to verify all project components
**Files added:** `verify_setup.py`
**Verification checks:**
1. Python packages (all 8 core dependencies)
2. .env file existence and required variables
3. PostgreSQL database connection
4. Gemini API key validity
5. NLTK punkt tokenizer data

**Usage:** `python verify_setup.py`

**Status:** ✅ Complete

### 2025-12-26 - Mission 1.11: Create .env.example Template ✅
**What we did:** Created `.env.example` template file
**Files added:** `.env.example`
**Template contents:**
- `PG_HOST`: PostgreSQL host (localhost)
- `PG_PORT`: PostgreSQL port (5433)
- `PG_DATABASE`: Database name (jeen_db)
- `PG_USER`: Database user (admin)
- `PG_PASSWORD`: Database password placeholder
- `GEMINI_API_KEY`: Google Gemini API key placeholder

**Status:** ✅ Complete

### 2025-12-26 - Mission 1.7 & 1.8: Create Database and Table Schema ✅
**What we did:** Created PostgreSQL database, user, and document_chunks table
**Database:** `jeen_db`
**User:** `admin`
**Table:** `document_chunks` with JSONB for embeddings

**Status:** ✅ Complete

### 2025-12-26 - Missions 1.1-1.5: Environment Setup ✅
**What we did:**
- Created `.gitignore` for sensitive files
- Created `requirements.txt` with 9 dependencies
- Created `setup_env.sh` automated setup script
- Initialized virtual environment
- Verified PostgreSQL installation (port 5433)

**Status:** ✅ Complete

---

## ✅ Completed Missions

- [x] Mission 1.1: Create .gitignore
- [x] Mission 1.2: Create requirements.txt
- [x] Mission 1.3: Create setup_env.sh
- [x] Mission 1.4: Run setup and initialize environment
- [x] Mission 1.5: Verify PostgreSQL Installation
- [x] Mission 1.7 & 1.8: Create Database and Table Schema
- [x] Mission 1.11: Create .env.example Template
- [x] Mission 1.13: Create Setup Verification Script
- [x] **BIG MISSION 2: Implement index_documents.py** ✅

---

## 🔐 Security Notes

- `.env` file contains sensitive credentials and is gitignored
- Never commit API keys or passwords to Git
- Use `.env.example` as a template for setting up credentials

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pypdf2 | 3.0.1 | PDF text extraction |
| python-docx | 1.1.0 | DOCX text extraction |
| nltk | 3.8.1 | Sentence tokenization |
| google-generativeai | 0.3.2 | Gemini API client |
| psycopg2-binary | 2.9.9 | PostgreSQL driver |
| pgvector | 0.2.4 | Vector operations |
| python-dotenv | 1.0.0 | Environment variables |
| tqdm | 4.66.1 | Progress bars |
| tenacity | 8.2.3 | Retry logic |

---

**Assignment:** Jeen AI Solutions - Phase 2
**Date Started:** December 26, 2025
