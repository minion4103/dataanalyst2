from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI()
load_dotenv()

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

# Add caching for prompt files
@functools.lru_cache(maxsize=10)
def read_prompt_file(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read()

async def ping_gemini(question_text, relevant_context="", max_tries=3):
    tries = 0
    while tries < max_tries:
        try:
            print(f"gemini is running {tries + 1} try")
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": gemini_api
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
                "model": "gpt-4.1-nano",
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

def extract_json_from_output(output: str) -> str:
    """Extract JSON from output that might contain extra text"""
    output = output.strip()
    
    json_patterns = [
        r'\[.*\]',  # Array pattern
        r'\{.*\}',  # Object pattern
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            return matches[0]
    
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
            df = result["dataframe"]
            
            if not df.empty:
                filename = f"data{i+1}.csv" if i > 0 else "data.csv"
                df.to_csv(filename, index=False, encoding="utf-8")
                
                scraped_data.append({
                    "filename": filename,
                    "source_url": url,
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample_data": df.head(3).to_dict('records'),
                    "description": f"Scraped data from {url}"
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
            
            # Get schema (column info)
            if format_type == "parquet" or "parquet" in url:
                schema_query = f"DESCRIBE SELECT * FROM read_parquet('{url}') LIMIT 0"
                sample_query = f"SELECT * FROM read_parquet('{url}') LIMIT 5"
            elif format_type == "csv" or "csv" in url:
                schema_query = f"DESCRIBE SELECT * FROM read_csv_auto('{url}') LIMIT 0"
                sample_query = f"SELECT * FROM read_csv_auto('{url}') LIMIT 5"
            elif format_type == "json" or "json" in url:
                schema_query = f"DESCRIBE SELECT * FROM read_json_auto('{url}') LIMIT 0"
                sample_query = f"SELECT * FROM read_json_auto('{url}') LIMIT 5"
            else:
                print(f"❌ Unsupported format: {format_type}")
                continue
            
            # Get schema
            schema_df = conn.execute(schema_query).fetchdf()
            schema_info = {
                "columns": list(schema_df['column_name']),
                "column_types": dict(zip(schema_df['column_name'], schema_df['column_type']))
            }
            
            # Get sample data
            sample_df = conn.execute(sample_query).fetchdf()
            
            database_info.append({
                "filename": f"database_{i+1}",
                "source_url": url,
                "format": format_type,
                "schema": schema_info,
                "sample_data": sample_df.head(3).to_dict('records'),
                "description": db_file.get("description", f"Database file ({format_type})"),
                "access_query": sample_query.replace("LIMIT 5", ""),  # Remove limit for actual queries
                "total_columns": len(schema_info["columns"])
            })
            
            print(f"✅ Database schema extracted: {len(schema_info['columns'])} columns")
            
        except Exception as e:
            print(f"❌ Failed to get schema for {db_file['url']}: {e}")
    
    conn.close()
    return database_info

def create_data_summary(csv_data: list, provided_csv_info: dict, database_info: list) -> dict:
    """Create comprehensive data summary for LLM code generation"""
    
    summary = {
        "provided_csv": None,
        "scraped_data": [],
        "database_files": [],
        "total_sources": 0
    }
    
    # Add provided CSV info
    if provided_csv_info:
        summary["provided_csv"] = provided_csv_info
        summary["total_sources"] += 1
    
    # Add scraped data
    summary["scraped_data"] = csv_data
    summary["total_sources"] += len(csv_data)
    
    # Add database info
    summary["database_files"] = database_info
    summary["total_sources"] += len(database_info)
    
    return summary

@app.post("/aianalyst/")
async def aianalyst(
    file: UploadFile = File(...),
    image: UploadFile = File(None),
    csv: UploadFile = File(None)
):
    time_start = time.time()
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

    # Step 6: Get database schemas and sample data
    database_info = []
    if extracted_sources.get('database_files'):
        database_info = await get_database_schemas(extracted_sources['database_files'])

    # Step 7: Create comprehensive data summary
    data_summary = create_data_summary(scraped_data, provided_csv_info, database_info)
    
    # Save data summary for debugging
    with open("data_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(data_summary), f, indent=2)

    print(f"📋 Data Summary: {data_summary['total_sources']} total sources")

    # Break down tasks
    task_breaker_instructions = read_prompt_file("prompts/task_breaker.txt")
    gemini_response = await ping_gemini(question_text, task_breaker_instructions)
    task_breaked = gemini_response["candidates"][0]["content"]["parts"][0]["text"]

    with open("broken_down_tasks.txt", "w", encoding="utf-8") as f:
        f.write(str(task_breaked))

    # Step 8: Generate final code based on all data sources
    # Use unified instructions that handle all source types
    code_instructions = read_prompt_file("prompts/unified_code_instructions.txt")

    context = (
        "ORIGINAL QUESTION: " + question_text + "\n\n" +
        "TASK BREAKDOWN: " + task_breaked + "\n\n" +
        "INSTRUCTIONS: " + code_instructions + "\n\n" +
        "DATA SUMMARY: " + json.dumps(make_json_serializable(data_summary), indent=2)
    )

    chatgpt_response = await ping_chatgpt(context , "You are a great Python code developer. Who write final code for the answer and our workflow using all the detial provided to you" )

    # Clean the code response (remove markdown formatting)
    raw_code = chatgpt_response["choices"][0]["message"]["content"]
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

    with open("chatgpt_code.py", "w") as f:
        f.write(cleaned_code)

    # Execute the code
    try:
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
                "CODE FIXING REQUIREMENTS:\n"
                "DO NOT MAKE ANY ASSUMPTIONS ABOUT THE CODE\n"
                "FIX THE CODE DO NOT GIVE ANSWER YOURSELF\n OR ANY EXPLANATION\n"
                "- The following Python code failed to run or didn't produce valid JSON output\n"
                "- DO NOT ADD markdown tags like ```python or ``` - send only raw code\n"
                "- Fix the errors but maintain the original logic and functionality\n"
                "- Return the complete fixed code, not just the problematic parts\n"
                "- Output MUST be valid JSON format (array [] or object {})\n"
                "- Use json.dumps() or ensure proper JSON formatting in print statements\n"
                "- Handle multiple data sources properly (CSV files and databases)\n"
                "- Install missing packages with pip if needed\n\n"
                
                "Data Summary: " + json.dumps(make_json_serializable(data_summary), indent=2) + "\n\n"
                f"Error Details and Code to Fix:\n{error_message}"
            )
            
            chatgpt_fix = await ping_chatgpt(fix_prompt, "You are a helpful Python code fixer.")
            fixed_code = chatgpt_fix["choices"][0]["message"]["content"]
            
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

            # Test the fixed code
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
    return {"error": "Code execution failed after all attempts", "time": time.time() - time_start}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)