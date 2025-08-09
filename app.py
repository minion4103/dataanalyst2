from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from matplotlib.style import context
import uvicorn
import base64
import httpx
from bs4 import BeautifulSoup
import time
import subprocess
import json
from dotenv import load_dotenv
import os
import data_scrape
import functools
import re
import pandas as pd
import numpy as np
from io import StringIO
from urllib.parse import urlparse
import duckdb
import glob

app = FastAPI()
load_dotenv()

# --- Precise file tracking & cleanup helpers ---
def _snapshot_files(root: str = ".") -> set[str]:
    """Get a snapshot of all files under root as relative paths."""
    files = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtual envs or cache folders commonly present
        parts = os.path.relpath(dirpath, root).split(os.sep)
        if any(p in {".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"} for p in parts):
            continue
        for fn in filenames:
            rel = os.path.normpath(os.path.join(os.path.relpath(dirpath, root), fn))
            files.add(rel)
    return files

def _cleanup_created_files(files_to_delete: set[str]) -> int:
    """Delete specific files created during this request.
    Returns number of files deleted."""
    deleted = 0
    for rel_path in files_to_delete:
        try:
            path = os.path.normpath(rel_path)
            # handle paths that might already be absolute
            if not os.path.isabs(path):
                path = os.path.join(".", path) if path != "." else "."
            if os.path.isfile(path):
                os.remove(path)
                deleted += 1
                print(f"🗑️ Deleted: {path}")
        except Exception as e:
            print(f"⚠️ Could not delete {rel_path}: {e}")
    print(f"🧹 Cleanup complete: {deleted} files deleted")
    return deleted

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
open_ai_url = "https://aipipe.org/openai/v1/chat/completions"
ocr_api_key = os.getenv("OCR_API_KEY")
OCR_API_URL = "https://api.ocr.space/parse/image"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
gemini_api = os.getenv("gemini_api")
horizon_api = os.getenv("horizon_api")
gemini_api_2 = os.getenv("gemini_api_2")
grok_api = os.getenv("grok_api")
grok_fix_api = os.getenv("grok_fix_api")

def make_json_serializable(obj):
    """Convert pandas/numpy objects to JSON-serializable formats"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series)):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype') and hasattr(obj, 'name'):
        return str(obj)
    elif pd.api.types.is_extension_array_dtype(obj):
        return str(obj)
    elif str(type(obj)).startswith("<class 'pandas."):
        return str(obj)
    elif str(type(obj)).startswith("<class 'numpy."):
        try:
            return obj.item() if hasattr(obj, 'item') else str(obj)
        except:
            return str(obj)
    else:
        return obj

# --- Safe file writing to avoid Windows cp1252 'charmap' UnicodeEncodeErrors ---
def safe_write(path: str, text: str, replace: bool = True):
    """Write text to file using UTF-8 regardless of system locale.

    Windows default (cp1252) cannot encode characters like U+2011 (non-breaking hyphen)
    or U+202F (narrow no-break space) sometimes produced by LLM outputs. This helper
    forces utf-8 and optionally replaces unencodable characters.
    """
    errors_policy = "replace" if replace else "strict"
    with open(path, "w", encoding="utf-8", errors=errors_policy) as f:
        f.write(text)

# Add caching for prompt files (with graceful fallback when missing)
@functools.lru_cache(maxsize=10)
def read_prompt_file(filename: str, default: str = "") -> str:
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ Prompt file not found: {filename}. Using default content.")
        return default

async def ping_gemini(question_text, relevant_context="", max_tries=3):
    tries = 0
    while tries < max_tries:
        if tries % 2 != 0:
            api_key = gemini_api
        else:
            api_key = gemini_api_2
        try:
            print(f"gemini is running {tries + 1} try")
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": relevant_context},
                            {"text": question_text}
                        ]
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(GEMINI_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error during Gemini call: {e}")
            tries += 1
    return {"error": "Gemini failed after max retries"}

async def ping_chatgpt(question_text, relevant_context, max_tries=3):
    tries = 0
    while tries < max_tries:
        try:
            print(f"openai is running {tries+1} try")
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": relevant_context},
                    {"role": "user", "content": question_text}
                ]
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(open_ai_url, headers=headers, json=payload)
                return response.json()
        except Exception as e:
            print(f"Error creating payload: {e}")
            tries += 1
            continue

async def ping_gemini_pro(question_text, relevant_context="", max_tries=3):
    """Call Gemini Pro API for code generation."""    
    tries = 0
    while tries < max_tries:
        if tries % 2 == 0:
            api_key = gemini_api
        else:
            api_key = gemini_api_2
        try:
            print(f"gemini pro is running {tries + 1} try")
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": relevant_context},
                            {"text": question_text}
                        ]
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post("https://generativelanguage.googleapis.com/v1/models/gemini-2.5-pro:generateContent", headers=headers, json=payload)
                print(response)
                return response.json()
        except Exception as e:
            print(f"Error creating payload: {e}")
            tries += 1

async def ping_grok(question_text, relevant_context="", max_tries=3):
    """Call Groq's OpenAI-compatible Responses API for code generation using gpt-oss-120b."""

    if not grok_api:
        return {"error": "GROQ_API_KEY not configured"}

    tries = 0
    while tries < max_tries:
        try:
            print(f"grok is running {tries + 1} try")
            headers = {
                "Authorization": f"Bearer {grok_api}",
                "Content-Type": "application/json"
            }

            # Structured messages for better instruction handling
            messages = []
            if relevant_context:
                messages.append({"role": "system", "content": relevant_context})
            messages.append({"role": "user", "content": question_text})

            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "temperature": 0.2,         # Low temp for deterministic code
                "tool_choice": "none",      # Avoids forcing search tools
            }

            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result

        except Exception as e:
            print(f"Error during Grok call: {e}")
            tries += 1

    return {"error": "Grok failed after max retries"}

async def fix_with_grok(question_text, relevant_context="", max_tries=3):
    """Call Groq's API (using grok_fix_api key) for code FIXING purposes.

    Mirrors ping_grok but authenticates with grok_fix_api so quota / routing can
    be separated for fix attempts.
    """

    if not grok_fix_api:
        return {"error": "grok_fix_api key not configured"}

    tries = 0
    while tries < max_tries:
        try:
            print(f"grok (fix) is running {tries + 1} try")
            headers = {
                "Authorization": f"Bearer {grok_fix_api}",
                "Content-Type": "application/json"
            }

            messages = []
            if relevant_context:
                messages.append({"role": "system", "content": relevant_context})
            messages.append({"role": "user", "content": question_text})

            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "temperature": 0.2,
                "tool_choice": "none",
            }

            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            print(f"Error during Grok fix call: {e}")
            tries += 1

    return {"error": "Grok fix failed after max retries"}

def extract_json_from_output(output: str) -> str:
    """Extract JSON from output that might contain extra text"""
    output = output.strip()
    
    # First try to find complete JSON objects (prioritize these)
    object_pattern = r'\{.*\}'
    object_matches = re.findall(object_pattern, output, re.DOTALL)
    
    # If we find JSON objects, return the longest one (most complete)
    if object_matches:
        longest_match = max(object_matches, key=len)
        return longest_match
    
    # Only if no objects found, look for arrays
    array_pattern = r'\[.*\]'
    array_matches = re.findall(array_pattern, output, re.DOTALL)
    
    if array_matches:
        longest_match = max(array_matches, key=len)
        return longest_match
    
    return output

def is_valid_json_output(output: str) -> bool:
    """Check if the output is valid JSON without trying to parse it"""
    output = output.strip()
    return (output.startswith('{') and output.endswith('}')) or (output.startswith('[') and output.endswith(']'))

async def extract_all_urls_and_databases(question_text: str) -> dict:
    """Extract all URLs for scraping and database files from the question"""
    
    extraction_prompt = f"""
    Analyze this question and extract ONLY the ACTUAL DATA SOURCES needed to answer the questions:
    
    QUESTION: {question_text}
    
    CRITICAL INSTRUCTIONS:
    1. Look for REAL, COMPLETE URLs that contain actual data (not example paths or documentation links)
    2. Focus on data sources that are DIRECTLY needed to answer the specific questions being asked
    3. IGNORE example paths like "year=xyz/court=xyz" - these are just structure examples, not real URLs
    4. IGNORE reference links that are just for context (like documentation websites)
    5. Only extract data sources that have COMPLETE, USABLE URLs/paths
    
    DATA SOURCE TYPES TO EXTRACT:
    - Complete S3 URLs with wildcards (s3://bucket/path/file.parquet)
    - Complete HTTP/HTTPS URLs to data APIs or files
    - Working database connection strings
    - Complete file paths that exist and are accessible
    
    DO NOT EXTRACT:
    - Example file paths (containing "xyz", "example", "sample")
    - Documentation or reference URLs that don't contain data
    - Incomplete paths or URL fragments
    - File structure descriptions that aren't actual URLs
    
    CONTEXT ANALYSIS:
    Read the question carefully. If it mentions a specific database with a working query example, 
    extract that. If it only shows file structure examples, don't extract those.
    
    Return a JSON object with:
    {{
        "scrape_urls": ["only URLs that need to be scraped for data to answer questions"],
        "database_files": [
            {{
                "url": "complete_working_database_url_or_s3_path",
                "format": "parquet|csv|json",
                "description": "what data this contains that helps answer the questions"
            }}
        ],
        "has_data_sources": true/false
    }}
    
    EXAMPLES:
    ✅ EXTRACT: "s3://bucket/data/file.parquet?region=us-east-1" (complete S3 URL)
    ✅ EXTRACT: "https://api.example.com/data.csv" (working data URL)
    ❌ IGNORE: "data/pdf/year=xyz/court=xyz/file.pdf" (example path with placeholders)
    ❌ IGNORE: "https://documentation-site.com/" (reference link, not data)
    
    Be very selective - only extract what is actually needed and usable.
    """
    
    response = await ping_gemini(extraction_prompt, "You are a data source extraction expert. Return only valid JSON.")
    try:
        # Check if response has error
        if "error" in response:
            print(f"❌ Gemini API error: {response['error']}")
            return extract_urls_with_regex(question_text)
        
        # Extract text from response
        if "candidates" not in response or not response["candidates"]:
            print("❌ No candidates in Gemini response")
            return extract_urls_with_regex(question_text)
        
        response_text = response["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Raw response text: {response_text}")
        
        # Try to extract JSON from response (sometimes it's wrapped in markdown)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.rfind("```")
            response_text = response_text[json_start:json_end].strip()
        
        print(f"Extracted JSON text: {response_text}")
        return json.loads(response_text)
        
    except Exception as e:
        print(f"URL extraction error: {e}")
        # Fallback to regex extraction
        return extract_urls_with_regex(question_text)
    

def extract_urls_with_regex(question_text: str) -> dict:
    """Fallback URL extraction using regex with context awareness"""
    scrape_urls = []
    database_files = []
    
    # Find all HTTP/HTTPS URLs
    url_pattern = r'https?://[^\s\'"<>]+'
    urls = re.findall(url_pattern, question_text)
    
    for url in urls:
        # Clean URL (remove trailing punctuation)
        clean_url = re.sub(r'[.,;)]+$', '', url)
        
        # Skip example/documentation URLs that don't contain actual data
        skip_patterns = [
            'example.com', 'documentation', 'github.com', 'docs.', 'help.',
            '/docs/', '/help/', '/guide/', '/tutorial/'
        ]
        
        if any(pattern in clean_url.lower() for pattern in skip_patterns):
            continue
        
        # Check if it's a database file
        if any(ext in clean_url.lower() for ext in ['.parquet', '.csv', '.json']):
            format_type = "parquet" if ".parquet" in clean_url else "csv" if ".csv" in clean_url else "json"
            database_files.append({
                "url": clean_url,
                "format": format_type,
                "description": f"Database file ({format_type})"
            })
        else:
            # Only add to scrape_urls if it looks like it contains data
            # Skip pure documentation/reference sites
            if not any(skip in clean_url.lower() for skip in ['ecourts.gov.in']):  # Add known reference sites
                scrape_urls.append(clean_url)
    
    # Find S3 paths - but only complete ones, not examples
    s3_pattern = r's3://[^\s\'"<>]+'
    s3_urls = re.findall(s3_pattern, question_text)
    for s3_url in s3_urls:
        # Skip example paths with placeholders
        if any(placeholder in s3_url for placeholder in ['xyz', 'example', '***', 'EXAMPLE']):
            continue
            
        clean_s3 = s3_url.split()[0]  # Take only the URL part
        if '?' in clean_s3:
            # Keep query parameters for S3 (they often contain important config)
            pass
        
        database_files.append({
            "url": clean_s3,
            "format": "parquet",
            "description": "S3 parquet file"
        })
    
    return {
        "scrape_urls": scrape_urls,
        "database_files": database_files,
        "has_data_sources": len(scrape_urls) > 0 or len(database_files) > 0
    }

async def scrape_all_urls(urls: list) -> list:
    """Scrape all URLs and save as data1.csv, data2.csv, etc."""
    scraped_data = []
    sourcer = data_scrape.ImprovedWebScraper()
    
    for i, url in enumerate(urls):
        try:
            print(f"🌐 Scraping URL {i+1}/{len(urls)}: {url}")
            
            # Create config for web scraping
            source_config = {
                "source_type": "web_scrape",
                "url": url,
                "data_location": "Web page data",
                "extraction_strategy": "scrape_web_table"
            }
            
            # Extract data
            result = await sourcer.extract_data(source_config)
            
            # Handle multiple tables
            if "tables" in result:
                tables = result["tables"]
                table_names = result["metadata"].get("table_names", [])
                
                for j, table_data in enumerate(tables):
                    df = table_data["dataframe"]
                    table_name = table_data["table_name"]
                    
                    if not df.empty:
                        # Create unique filename with table name and index
                        safe_table_name = table_name.replace(" ", "_").replace("-", "_")
                        # Remove any problematic characters for filenames
                        safe_table_name = "".join(c for c in safe_table_name if c.isalnum() or c in ["_", "-"])
                        
                        if i == 0:  # First URL
                            filename = f"{safe_table_name}_{j+1}.csv"
                        else:  # Subsequent URLs
                            filename = f"{safe_table_name}_url{i+1}_{j+1}.csv"
                        
                        df.to_csv(filename, index=False, encoding="utf-8")
                        
                        scraped_data.append({
                            "filename": filename,
                            "source_url": url,
                            "table_name": table_name,
                            "shape": table_data["shape"],
                            "columns": table_data["columns"],
                            "sample_data": df.head(3).to_dict('records') if not df.empty else []
                        })
                        
                        print(f"💾 Saved {table_name} as {filename}")
            
            # Fallback for old single table format
            elif "dataframe" in result:
                df = result["dataframe"]
                
                if not df.empty:
                    filename = f"data{i+1}.csv" if i > 0 else "data.csv"
                    df.to_csv(filename, index=False, encoding="utf-8")
                    
                    scraped_data.append({
                        "filename": filename,
                        "source_url": url,
                        "shape": df.shape,
                        "columns": list(df.columns)
                    })
                
                print(f"✅ Saved {filename}: {df.shape} rows")
            else:
                print(f"⚠️ No data extracted from {url}")
                
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
    
    return scraped_data

async def get_database_schemas(database_files: list) -> list:
    """Get schema and sample data from database files without loading full data"""
    database_info = []
    
    # Setup DuckDB
    conn = duckdb.connect()
    try:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL parquet; LOAD parquet;")
        print("✅ DuckDB extensions loaded")
    except Exception as e:
        print(f"Warning: Could not load DuckDB extensions: {e}")
    
    for i, db_file in enumerate(database_files):
        try:
            url = db_file["url"]
            format_type = db_file["format"]
            
            print(f"📊 Getting schema for database {i+1}/{len(database_files)}: {url}")
            
            # Build lightweight FROM/SELECT SQL and schema query (no data loading)
            if format_type == "parquet" or "parquet" in url:
                from_clause = f"read_parquet('{url}')"
                base_select = f"SELECT * FROM {from_clause}"
                schema_query = f"DESCRIBE SELECT * FROM {from_clause} LIMIT 0"
            elif format_type == "csv" or "csv" in url:
                # Use small SAMPLE_SIZE to keep inference light
                from_clause = f"read_csv_auto('{url}', SAMPLE_SIZE=2048)"
                base_select = f"SELECT * FROM {from_clause}"
                schema_query = f"DESCRIBE SELECT * FROM {from_clause} LIMIT 0"
            elif format_type == "json" or "json" in url:
                from_clause = f"read_json_auto('{url}')"
                base_select = f"SELECT * FROM {from_clause}"
                schema_query = f"DESCRIBE SELECT * FROM {from_clause} LIMIT 0"
            else:
                print(f"❌ Unsupported format: {format_type}")
                continue
            
            # Get schema
            schema_df = conn.execute(schema_query).fetchdf()
            schema_info = {
                "columns": list(schema_df['column_name']),
                "column_types": dict(zip(schema_df['column_name'], schema_df['column_type']))
            }

            # Attempt to fetch a tiny sample (3 rows) for user visibility
            sample_data = []
            try:
                sample_query = f"{base_select} LIMIT 3"
                sample_df = conn.execute(sample_query).fetchdf()
                if not sample_df.empty:
                    # Convert to list[dict] keeping primitive types
                    sample_data = json.loads(sample_df.head(3).to_json(orient="records"))
            except Exception as sample_err:
                print(f"⚠️ Could not fetch sample rows for {url}: {sample_err}")

            database_info.append({
                "filename": f"database_{i+1}",
                "source_url": url,
                "format": format_type,
                "schema": schema_info,
                "description": db_file.get("description", f"Database file ({format_type})"),
                # Provide SQL strings to be used directly in DuckDB queries (do not execute here)
                "access_query": base_select,  # kept for backward compatibility
                "from_clause": from_clause,
                "preview_limit_sql": f"{base_select} LIMIT 10",
                "sample_data": sample_data,
                "total_columns": len(schema_info["columns"])
            })

            print(f"✅ Database schema extracted: {len(schema_info['columns'])} columns; sample_rows={len(sample_data)}")
            
        except Exception as e:
            print(f"❌ Failed to get schema for {db_file['url']}: {e}")
    
    conn.close()
    return database_info

def create_data_summary(csv_data: list, provided_csv_info: dict, database_info: list) -> dict:
    """Create comprehensive data summary for LLM code generation.
    Ensures total_sources counts unique sources across categories (no double counting)."""

    summary = {
        "provided_csv": None,
        "scraped_data": [],
        "database_files": [],
        "total_sources": 0,
    }

    # Add provided CSV info
    if provided_csv_info:
        summary["provided_csv"] = provided_csv_info

    # Add scraped data
    summary["scraped_data"] = csv_data

    # Add database info
    summary["database_files"] = database_info

    # Compute unique total sources by identifiers (filenames/URLs)
    identifiers = set()
    if provided_csv_info and provided_csv_info.get("filename"):
        identifiers.add(os.path.normpath(provided_csv_info["filename"]))
    for item in csv_data or []:
        fn = item.get("filename")
        if fn:
            identifiers.add(os.path.normpath(fn))
    for item in database_info or []:
        src = item.get("source_url") or item.get("filename")
        if src:
            # Normalize only path-like strings; URLs can be left as-is
            try:
                norm = os.path.normpath(src) if not (src.startswith("http://") or src.startswith("https://") or src.startswith("s3://")) else src
            except Exception:
                norm = src
            identifiers.add(norm)

    summary["total_sources"] = len(identifiers)
    return summary

@app.post("/aianalyst/")
async def aianalyst(
    file: UploadFile = File(...),
    image: UploadFile = File(None),
    csv: UploadFile = File(None)
):
    time_start = time.time()
    # Track files created during this request
    initial_snapshot = _snapshot_files(".")
    created_files: set[str] = set()
    content = await file.read()
    question_text = content.decode("utf-8")

    # Handle image if provided (existing logic)
    if image:
        try:
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            if not ocr_api_key:
                print("⚠️ OCR_API_KEY not found - skipping image processing")
                question_text += "\n\nOCR API key not configured - image text extraction skipped"
            else:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    form_data = {
                        "base64Image": f"data:image/png;base64,{base64_image}",
                        "apikey": ocr_api_key,
                        "language": "eng",
                        "scale": "true",
                        "OCREngine": "1"
                    }
                    
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    
                    response = await client.post(OCR_API_URL, data=form_data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if not result.get('IsErroredOnProcessing', True):
                            parsed_results = result.get('ParsedResults', [])
                            if parsed_results:
                                image_text = parsed_results[0].get('ParsedText', '').strip()
                                if image_text:
                                    question_text += f"\n\nExtracted from image:\n{image_text}"
                                    print("✅ Text extracted from image")
                    else:
                        print(f"❌ OCR API error: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error extracting text from image: {e}")

    # Step 3: Handle provided CSV file
    # EARLY TASK BREAKDOWN (user request: generate first before other heavy steps)
    # We do this after potential image OCR so the extracted text is included.
    task_breaker_instructions = read_prompt_file(
        "prompts/task_breaker.txt",
        default=(
            "You are a precise task breaker. Given a user question, output a concise, ordered list of actionable steps "
            "to analyze the data sources provided (CSV, scraped tables, or DuckDB FROM clauses). Keep steps specific "
            "(load data, validate schema, compute metrics, create plots, return final JSON)."
        ),
    )
    try:
        gemini_response = await ping_gemini(question_text, task_breaker_instructions)
        task_breaked = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        task_breaked = f"1. Read question (Task breaker fallback due to error: {e})"  # fallback minimal content
    with open("broken_down_tasks.txt", "w", encoding="utf-8") as f:
        f.write(str(task_breaked))
    created_files.add(os.path.normpath("broken_down_tasks.txt"))

    # Proceed with remaining steps (CSV processing, source extraction, etc.)
    # ----------------------------------------------------------------------
    provided_csv_info = None
    if csv:
        try:
            csv_content = await csv.read()
            csv_df = pd.read_csv(StringIO(csv_content.decode("utf-8")))
            
            # Clean the CSV
            sourcer = data_scrape.ImprovedWebScraper()
            cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(csv_df)
            
            # Save as ProvidedCSV.csv
            cleaned_df.to_csv("ProvidedCSV.csv", index=False, encoding="utf-8")
            created_files.add(os.path.normpath("ProvidedCSV.csv"))
            
            provided_csv_info = {
                "filename": "ProvidedCSV.csv",
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "sample_data": cleaned_df.head(3).to_dict('records'),
                "description": "User-provided CSV file (cleaned and formatted)",
                "formatting_applied": formatting_results
            }
            
            print(f"📝 Provided CSV processed: {cleaned_df.shape} rows, saved as ProvidedCSV.csv")
            
        except Exception as e:
            print(f"❌ Error processing provided CSV: {e}")

    # Step 4: Extract all URLs and database files from question
    print("🔍 Extracting all data sources from question...")
    extracted_sources = await extract_all_urls_and_databases(question_text)
    
    print(f"📊 Found {len(extracted_sources.get('scrape_urls', []))} URLs to scrape")
    print(f"📊 Found {len(extracted_sources.get('database_files', []))} database files")

    # Step 5: Scrape all URLs and save as CSV files
    scraped_data = []
    if extracted_sources.get('scrape_urls'):
        scraped_data = await scrape_all_urls(extracted_sources['scrape_urls'])
        # register scraped CSVs reported by the scraper
        for item in scraped_data:
            fn = item.get("filename")
            if fn:
                created_files.add(os.path.normpath(fn))

    # Step 6: Get database schemas and sample data
    # Build list of database files to process, prioritizing the uploaded CSV if present
    database_info = []
    database_files_to_process = []

    # If a CSV was uploaded, include it for schema extraction first
    if provided_csv_info:
        database_files_to_process.append({
            "url": "ProvidedCSV.csv",
            "format": "csv",
            "description": "User-provided CSV file (cleaned and formatted)",
        })

    # Extend with extracted database files, but skip nonexistent local files
    extracted_db_files = extracted_sources.get('database_files', []) or []

    def _looks_like_url(u: str) -> bool:
        return isinstance(u, str) and (u.startswith("http://") or u.startswith("https://") or u.startswith("s3://"))

    for db in extracted_db_files:
        try:
            url = db.get("url")
            fmt = db.get("format", "csv")
            if not url:
                continue
            if _looks_like_url(url):
                database_files_to_process.append({"url": url, "format": fmt, "description": db.get("description", f"Database file ({fmt})")})
            else:
                # Local path: include only if it exists to avoid errors like sample-sales.csv
                if os.path.exists(url):
                    database_files_to_process.append({"url": url, "format": fmt, "description": db.get("description", f"Database file ({fmt})")})
                else:
                    print(f"⏭️ Skipping nonexistent local database file: {url}")
        except Exception as _e:
            # If anything odd occurs, just skip this entry
            print(f"⏭️ Skipping invalid database file entry: {db}")

    if database_files_to_process:
        print(f"📊 Will process {len(database_files_to_process)} database files for schema extraction")
        database_info = await get_database_schemas(database_files_to_process)

    # Step 7: Create comprehensive data summary
    data_summary = create_data_summary(scraped_data, provided_csv_info, database_info)
    
    # Save data summary for debugging
    with open("data_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(data_summary), f, indent=2)
    created_files.add(os.path.normpath("data_summary.json"))

    print(f"📋 Data Summary: {data_summary['total_sources']} total sources")

    # Step 8: Generate final code based on all data sources
    # Use unified instructions that handle all source types
    code_instructions = read_prompt_file(
        "prompts/unified_code_instructions.txt",
        default=(
            "Write a single self-contained Python script that: (1) Uses only the data sources listed in DATA SUMMARY; "
            "(2) Loads CSVs directly from local paths provided (e.g., ProvidedCSV.csv) or uses DuckDB FROM clauses "
            "for remote sources; (3) Performs the requested computations/plots; (4) Prints ONLY a valid JSON object "
            "to stdout via json.dumps with final results; (5) Do not access any placeholder files or URLs not in the "
            "DATA SUMMARY; (6) Import required libraries; (7) Avoid interactive UI."
        ),
    )

    context = (
        "ORIGINAL QUESTION: " + question_text + "\n\n" +
        "TASK BREAKDOWN: " + task_breaked + "\n\n" +
        "INSTRUCTIONS: " + code_instructions + "\n\n" +
        "DATA SUMMARY: " + json.dumps(make_json_serializable(data_summary), indent=2)
    )

    # horizon_response = await ping_chatgpt(context, "You are a great Python code developer.JUST GIVE CODE NO EXPLANATIONS Who write final code for the answer and our workflow using all the detail provided to you")
    # horizon_response = await ping_grok(context, "You are a great Python code developer.JUST GIVE CODE NO EXPLANATIONS Who write final code for the answer and our workflow using all the detail provided to you")
    # Validate Grok response structure before trying to index
    gemini_response = await ping_gemini_pro(context, "You are a great Python code developer. JUST GIVE CODE NO EXPLANATIONS. Write final code for the answer and our workflow using all the detail provided to you")
    raw_code = gemini_response["candidates"][0]["content"]["parts"][0]["text"]

    
    lines = raw_code.split('\n')
    clean_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block or (not line.strip().startswith('```') and '```' not in line):
            clean_lines.append(line)

    cleaned_code = '\n'.join(clean_lines).strip()

    # Write generated code using UTF-8 to avoid Windows cp1252 encode errors (e.g. for narrow no-break space \u202f)
    with open("chatgpt_code.py", "w", encoding="utf-8", errors="replace") as f:
        f.write(cleaned_code)
    created_files.add(os.path.normpath("chatgpt_code.py"))

    # Execute the code
    try:
        # Snapshot before executing generated code to catch any new files it creates
        pre_exec_snapshot = _snapshot_files(".")
        result = subprocess.run(
            ["python", "chatgpt_code.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            stdout = result.stdout.strip()
            json_output = extract_json_from_output(stdout)
            
            if is_valid_json_output(json_output):
                try:
                    output_data = json.loads(json_output)
                    print("✅ Code executed successfully")
                    
                    # Cleanup generated files before returning
                    post_exec_snapshot = _snapshot_files(".")
                    new_files = post_exec_snapshot - pre_exec_snapshot
                    files_to_delete = {os.path.normpath(p) for p in new_files} | created_files
                    _cleanup_created_files(files_to_delete)
                    
                    return output_data
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {str(e)[:100]}")
            else:
                print(f"Output doesn't look like JSON: {json_output[:100]}")
        else:
            print(f"Execution error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Code execution timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Code fixing attempts (existing logic)
    max_fix_attempts = 3
    fix_attempt = 0
    
    while fix_attempt < max_fix_attempts:
        fix_attempt += 1
        print(f"🔧 Attempting to fix code (attempt {fix_attempt}/{max_fix_attempts})")
        
        try:
            with open("chatgpt_code.py", "r", encoding="utf-8") as code_file:
                code_content = code_file.read()
            
            try:
                # Snapshot for this fix attempt
                fix_pre_exec_snapshot = _snapshot_files(".")
                result = subprocess.run(
                    ["python", "chatgpt_code.py"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                error_context = f"Return code: {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            except Exception as e:
                error_context = f"Execution failed with exception: {str(e)}"
            
            error_message = f"Error: {error_context}\n\nCode:\n{code_content}\n\nTask breakdown:\n{task_breaked}"
            
            fix_prompt = (
                "URGENT CODE FIXING TASK: CURRENT BROKEN CODE: " + str(cleaned_code) + "\n" + 
                "ERROR DETAILS: " + str(error_message) + "\n" +
                "AVAILABLE DATA (use these exact sources): " + str(data_summary) + "\n\n" +
                "FIXING INSTRUCTIONS:\n" +
                "1. Fix the specific error mentioned above\n" +
                "2. Use ONLY the data sources listed in AVAILABLE DATA section\n" +
                "3. DO NOT add placeholder URLs or fake data\n" +
                "Instead:\n" +
                "                    Use DATEDIFF('day', start_date, end_date) for number of days.\n" +
                "\n" +
                "                    Or use date_part() only on actual DATE/TIMESTAMP/INTERVAL types.\n" +
                "\n" +
                "                    Always check the DuckDB function signature before applying a function.\n" +
                "                    If a function call results in a type mismatch, either cast to the required type or choose an alternative function that directly returns the needed value."
                "4. DO NOT create imaginary answers - process actual data\n" +
                "5. Ensure final output is valid JSON using json.dumps()\n" +
                "6. Make the code complete and executable\n\n" +
                "COMMON FIXES NEEDED:\n" +
                "- Replace placeholder URLs with actual ones from data_summary\n" +
                "- Fix file path references to match available files\n" +
                "- Add missing imports\n" +
                "- Fix syntax errors\n" +
                "- Ensure proper JSON output format\n\n" +
                "Return ONLY the corrected Python code (no markdown, no explanations):"
            )
            # Write fix prompt safely (avoid cp1252 encoding errors on Windows)
            safe_write("fix.txt", fix_prompt)

            # horizon_fix = await fix_with_grok(fix_prompt, "You are a helpful Python code fixer. dont try to code from scratch. just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            # fixed_code = horizon_fix["choices"][0]["message"]["content"]
            gemini_fix = await ping_gemini_pro(fix_prompt, "You are a helpful Python code fixer. Don't try to code from scratch. Just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            fixed_code = gemini_fix["candidates"][0]["content"]["parts"][0]["text"]

            # Clean the fixed code
            lines = fixed_code.split('\n')
            clean_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not line.strip().startswith('```') and '```' not in line):
                    clean_lines.append(line)

            cleaned_fixed_code = '\n'.join(clean_lines).strip()
            
            with open("chatgpt_code.py", "w", encoding="utf-8") as code_file:
                code_file.write(cleaned_fixed_code)
            created_files.add(os.path.normpath("chatgpt_code.py"))

            # Test the fixed code
            # Track any new files produced by retries as well
            result = subprocess.run(
                ["python", "chatgpt_code.py"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                stdout = result.stdout.strip()
                json_output = extract_json_from_output(stdout)
                
                if is_valid_json_output(json_output):
                    try:
                        output_data = json.loads(json_output)
                        print(f"✅ Code fixed and executed successfully on fix attempt {fix_attempt}")
                        
                        # Cleanup generated files before returning
                        post_exec_snapshot = _snapshot_files(".")
                        # Prefer fix attempt snapshot if present
                        new_files = post_exec_snapshot - (fix_pre_exec_snapshot if 'fix_pre_exec_snapshot' in locals() else pre_exec_snapshot)
                        files_to_delete = {os.path.normpath(p) for p in new_files} | created_files
                        
                        _cleanup_created_files(files_to_delete)
                        return output_data
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error on fix attempt {fix_attempt}: {str(e)[:100]}")
                else:
                    print(f"Output still doesn't look like JSON on fix attempt {fix_attempt}: {json_output[:100]}")
            else:
                print(f"Execution still failing on fix attempt {fix_attempt}: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"Code execution timed out on fix attempt {fix_attempt}")
        except Exception as e:
            print(f"Unexpected error on fix attempt {fix_attempt}: {e}")

    # If all attempts fail
    print("❌ All code execution attempts failed")
    
    # Cleanup generated files before returning error
    final_snapshot = _snapshot_files(".")
    new_files = final_snapshot - initial_snapshot
    files_to_delete = {os.path.normpath(p) for p in new_files} | created_files

    _cleanup_created_files(files_to_delete)
    return {"error": "Code execution failed after all attempts", "time": time.time() - time_start}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)