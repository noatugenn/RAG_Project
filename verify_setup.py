#!/usr/bin/env python3
"""
verify_setup.py - Setup Verification Script for Jeen Project

This script verifies that all components are properly configured:
1. Python packages are installed
2. .env file exists with required variables
3. PostgreSQL database connection works
4. Gemini API key is valid
5. Database table exists

Run this before using index_documents.py to ensure everything is set up correctly.
"""

import sys
import os

def print_status(name, success, message=""):
    """Print a status line with checkmark or X"""
    status = "✓" if success else "✗"
    color_start = "\033[92m" if success else "\033[91m"  # Green or Red
    color_end = "\033[0m"
    print(f"  {color_start}{status}{color_end} {name}", end="")
    if message:
        print(f" - {message}")
    else:
        print()

def check_python_packages():
    """Check if all required Python packages are installed"""
    print("\n[1/5] Checking Python packages...")

    packages = {
        'PyPDF2': 'pypdf2',
        'docx': 'python-docx',
        'nltk': 'nltk',
        'google.generativeai': 'google-generativeai',
        'psycopg2': 'psycopg2-binary',
        'dotenv': 'python-dotenv',
        'tqdm': 'tqdm',
        'tenacity': 'tenacity'
    }

    all_ok = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print_status(package_name, True)
        except ImportError:
            print_status(package_name, False, "Not installed")
            all_ok = False

    return all_ok

def check_env_file():
    """Check if .env file exists and has required variables"""
    print("\n[2/5] Checking .env file...")

    env_path = os.path.join(os.path.dirname(__file__), '.env')

    if not os.path.exists(env_path):
        print_status(".env file", False, "File not found")
        return False

    print_status(".env file", True, "File exists")

    # Load .env file
    from dotenv import load_dotenv
    load_dotenv(env_path)

    required_vars = ['PG_HOST', 'PG_PORT', 'PG_DATABASE', 'PG_USER', 'PG_PASSWORD', 'GEMINI_API_KEY']
    all_ok = True

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'PASSWORD' in var or 'KEY' in var:
                masked = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
                print_status(var, True, f"Set ({masked})")
            else:
                print_status(var, True, f"Set ({value})")
        else:
            print_status(var, False, "Not set")
            all_ok = False

    return all_ok

def check_database_connection():
    """Check if PostgreSQL database connection works"""
    print("\n[3/5] Checking PostgreSQL connection...")

    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        host = os.getenv('PG_HOST', 'localhost')
        port = os.getenv('PG_PORT', '5433')
        database = os.getenv('PG_DATABASE', 'jeen_db')
        user = os.getenv('PG_USER', 'admin')
        password = os.getenv('PG_PASSWORD', '')

        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

        print_status("Connection", True, f"Connected to {database}@{host}:{port}")

        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'document_chunks'
            );
        """)
        table_exists = cursor.fetchone()[0]

        if table_exists:
            # Count rows
            cursor.execute("SELECT COUNT(*) FROM document_chunks;")
            count = cursor.fetchone()[0]
            print_status("document_chunks table", True, f"Exists ({count} rows)")
        else:
            print_status("document_chunks table", False, "Table not found")

        cursor.close()
        conn.close()
        return True

    except psycopg2.OperationalError as e:
        print_status("Connection", False, str(e).split('\n')[0])
        return False
    except Exception as e:
        print_status("Connection", False, str(e))
        return False

def check_gemini_api():
    """Check if Gemini API key is valid"""
    print("\n[4/5] Checking Gemini API...")

    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print_status("API Key", False, "GEMINI_API_KEY not set")
            return False

        genai.configure(api_key=api_key)

        # Try to generate a simple embedding to test the API
        result = genai.embed_content(
            model="models/embedding-001",
            content="Test",
            task_type="retrieval_document"
        )

        embedding_size = len(result['embedding'])
        print_status("API Key", True, "Valid")
        print_status("Embedding model", True, f"models/embedding-001 ({embedding_size} dimensions)")
        return True

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            print_status("API Key", False, "Invalid API key")
        else:
            print_status("API Key", False, error_msg[:50])
        return False

def check_nltk_data():
    """Check if NLTK punkt tokenizer is downloaded"""
    print("\n[5/5] Checking NLTK data...")

    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print_status("punkt tokenizer", True, "Downloaded")
        return True
    except LookupError:
        print_status("punkt tokenizer", False, "Not downloaded")
        print("       Run: python -c \"import nltk; nltk.download('punkt')\"")
        return False

def main():
    """Run all verification checks"""
    print("=" * 50)
    print("  Jeen Project - Setup Verification")
    print("=" * 50)

    results = []

    # Run all checks
    results.append(("Python packages", check_python_packages()))
    results.append(("Environment file", check_env_file()))
    results.append(("Database connection", check_database_connection()))
    results.append(("Gemini API", check_gemini_api()))
    results.append(("NLTK data", check_nltk_data()))

    # Summary
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        print_status(name, passed)
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("\033[92m✓ All checks passed! Ready to run index_documents.py\033[0m")
        return 0
    else:
        print("\033[91m✗ Some checks failed. Please fix the issues above.\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())
