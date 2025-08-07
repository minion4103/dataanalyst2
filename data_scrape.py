import pandas as pd
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Optional, Any
import numpy as np
import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
import httpx
import os
from dotenv import load_dotenv
from io import StringIO

load_dotenv()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
gemini_api = os.getenv("gemini_api")

async def ping_gemini(question_text, relevant_context="", max_tries=3):
    tries = 0
    while tries < max_tries:
        try:
            print(f"gemini is running {tries + 1} try")
            
            # Check if API key is available
            if not gemini_api:
                print("❌ Gemini API key not found in environment variables")
                return {"error": "Gemini API key not configured"}
            
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
                
                # Debug: Print response content
                response_text = response.text
                print(f"Gemini response length: {len(response_text)}")
                
                if not response_text.strip():
                    raise Exception("Empty response from Gemini API")
                
                try:
                    return response.json()
                except json.JSONDecodeError as json_error:
                    print(f"JSON decode error: {json_error}")
                    print(f"Response content: {response_text[:500]}...")
                    raise Exception(f"Invalid JSON response: {json_error}")
                    
        except Exception as e:
            print(f"Error during Gemini call: {e}")
            tries += 1
    return {"error": "Gemini failed after max retries"}

class NumericFieldFormatter:
    """Handles identification and cleaning of numeric fields in DataFrames"""
    
    def __init__(self):
        self.currency_symbols = ['$', '€', '£', '¥', '₹', '₽', 'R$', 'A$', 'C$', '₦', '₨']
        self.percentage_indicators = ['%']
    
    async def identify_numeric_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Use Gemini to identify which columns should be numeric and their types"""
        
        # Get sample data for analysis
        sample_data = []
        for col in df.columns:
            sample_values = df[col].dropna().head(10)
            
            # Convert non-serializable types to strings
            serializable_values = []
            for val in sample_values:
                if pd.api.types.is_datetime64_any_dtype(type(val)) or hasattr(val, 'strftime'):
                    # Convert datetime/timestamp to string
                    serializable_values.append(str(val))
                elif hasattr(val, 'item'):
                    # Convert numpy types to Python native types
                    serializable_values.append(val.item())
                else:
                    serializable_values.append(val)
            
            sample_data.append({
                "column_name": col,
                "sample_values": serializable_values,
                "current_dtype": str(df[col].dtype)
            })
        
        # Skip datetime columns from numeric formatting
        datetime_columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                datetime_columns.append(col)
        
        identification_prompt = f"""
        Analyze these DataFrame columns and identify which ones contain NUMERIC DATA that needs cleaning:
        
        Column Data: {json.dumps(sample_data, indent=2)}
        
        Note: Skip these datetime columns from numeric formatting: {datetime_columns}
        
        Look for columns that contain:
        1. Currency values (with symbols like $, €, £, ¥, etc.)
        2. Percentages (with % symbol)
        3. Numbers with formatting (commas, spaces, brackets)
        4. Scientific notation (1.23e+05)
        5. Mixed text-numeric values where numeric part can be extracted
        6. Integer or float values that need type conversion
        
        IMPORTANT: Only identify columns that actually contain NUMERIC DATA, even if formatted as text.
        DO NOT mark columns as numeric if they contain:
        - Pure text/names/categories
        - IDs/codes that are meant to stay as text
        - Dates/timestamps (already excluded)
        - Yes/No or True/False values
        
        Return a JSON object with this structure:
        {{
            "column_name": {{
                "is_numeric": true/false,
                "numeric_type": "currency" | "percentage" | "integer" | "float" | "scientific",
                "target_dtype": "int64" | "float64",
                "cleaning_needed": true/false,
                "confidence": "high" | "medium" | "low",
                "description": "brief description of why this column is/isn't numeric"
            }}
        }}
        
        Examples of what SHOULD be identified as numeric:
        - ["$1,234.56", "$2,000", "€500"] → currency
        - ["45%", "12.5%", "100%"] → percentage  
        - ["1.23e+05", "2.5E-03"] → scientific notation
        - ["1,234,567", "2,000", "500"] → integer with formatting
        - ["T$2,257,844", "F8$1,238"] → currency (extract numeric part)
        
        Examples of what should NOT be identified as numeric:
        - ["Product A", "Category B", "Name"] → text
        - ["ID001", "USER123", "CODE456"] → text IDs
        - ["Yes", "No", "Maybe"] → categorical
        - ["2023-01-01", "12:30:00"] → dates/times (already excluded)
        """
        
        response = await ping_gemini(identification_prompt, "You are a data analysis expert specializing in numeric data identification. Return only valid JSON.")
        
        try:
            # Check if response has error
            if "error" in response:
                print(f"❌ Gemini API error: {response['error']}")
                print("🔄 Falling back to heuristic identification...")
                return self._fallback_numeric_identification(df)
            
            # Extract text from response
            if "candidates" not in response or not response["candidates"]:
                print("❌ No candidates in Gemini response")
                print("🔄 Falling back to heuristic identification...")
                return self._fallback_numeric_identification(df)
            
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            print(f"Gemini response text length: {len(response_text)}")
            
            # Try to extract JSON from response (sometimes it's wrapped in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            analysis = json.loads(response_text)
            # Filter out datetime columns and non-numeric columns from analysis
            filtered_analysis = {}
            for col, info in analysis.items():
                if col not in datetime_columns and info.get("is_numeric", False):
                    filtered_analysis[col] = info
            
            print(f"✅ LLM identified {len(filtered_analysis)} numeric columns: {list(filtered_analysis.keys())}")
            return filtered_analysis
        except Exception as e:
            print(f"❌ Error in Gemini numeric analysis: {e}")
            print("🔄 Falling back to heuristic identification...")
            # Fallback to existing heuristic method
            return self._fallback_numeric_identification(df)
    
    def _fallback_numeric_identification(self, df: pd.DataFrame) -> Dict[str, str]:
        """Fallback method to identify numeric columns using heuristics"""
        numeric_columns = {}
        
        for col in df.columns:
            # Skip datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                continue
                
            sample_values = df[col].dropna().astype(str).head(20).tolist()
            
            # Check if most values look numeric
            numeric_count = 0
            for val in sample_values:
                if self._looks_numeric(val):
                    numeric_count += 1
            
            if len(sample_values) > 0 and numeric_count / len(sample_values) > 0.7:  # 70% threshold
                numeric_type = self._detect_numeric_type(sample_values)
                numeric_columns[col] = {
                    "is_numeric": True,
                    "numeric_type": numeric_type,
                    "target_dtype": "float64" if numeric_type in ["currency", "percentage", "float"] else "int64",
                    "cleaning_needed": True,
                    "description": f"Auto-detected {numeric_type} column"
                }
        
        return numeric_columns
    
    def _looks_numeric(self, value: str) -> bool:
        """Check if a string value looks like it could be numeric"""
        # Remove common non-numeric characters
        cleaned = re.sub(r'[,$%€£¥₹₽\s\[\]#\-TFRK]', '', str(value))
        
        # Check if what remains is mostly digits and decimal points
        if not cleaned:
            return False
            
        return bool(re.match(r'^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', cleaned))
    
    def _detect_numeric_type(self, sample_values: List[str]) -> str:
        """Detect the type of numeric data"""
        sample_str = ' '.join(sample_values)
        
        if any(symbol in sample_str for symbol in self.currency_symbols):
            return "currency"
        elif '%' in sample_str:
            return "percentage"
        elif 'e' in sample_str.lower() or 'E' in sample_str:
            return "scientific"
        elif '.' in sample_str:
            return "float"
        else:
            return "integer"
    
    def clean_numeric_column(self, series: pd.Series, numeric_info: Dict[str, Any]) -> pd.Series:
        """Clean a single numeric column based on its identified type"""
        numeric_type = numeric_info.get("numeric_type", "float")
        target_dtype = numeric_info.get("target_dtype", "float64")
        
        print(f"Cleaning column as {numeric_type} -> {target_dtype}")
        
        # Convert to string for cleaning
        cleaned_series = series.astype(str)
        
        if numeric_type == "currency":
            cleaned_series = self._clean_currency_column(cleaned_series)
        elif numeric_type == "percentage":
            cleaned_series = self._clean_percentage_column(cleaned_series)
        elif numeric_type == "scientific":
            cleaned_series = self._clean_scientific_column(cleaned_series)
        else:
            cleaned_series = self._clean_generic_numeric_column(cleaned_series)
        
        # Convert to target dtype
        try:
            if target_dtype == "int64":
                # First convert to float to handle decimals, then to int
                cleaned_series = pd.to_numeric(cleaned_series, errors='coerce').fillna(0).astype('int64')
            else:
                cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert to {target_dtype}, keeping as float64: {e}")
            cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')
        
        return cleaned_series
    
    def _clean_currency_column(self, series: pd.Series) -> pd.Series:
        """Clean currency values with improved handling of complex prefixes"""
        def clean_currency_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val).strip()
            if not val_str:
                return np.nan
            
            # Remove quotes if present
            val_str = val_str.strip('"\'')
            
            # Handle complex prefixes like "T$2,257,844,554", "F8$1,238,764,765", "DKR$1,081,169,825", "4TS3", "24RK", etc.
            
            # Step 1: Try to extract just the numeric part with $ and commas
            # Look for patterns like $X,XXX,XXX,XXX or variations with prefixes
            numeric_patterns = [
                r'\$[\d,]+(?:\.\d+)?',  # Standard $X,XXX,XXX format
                r'[\d,]+(?:\.\d+)?',    # Just numbers with commas
                r'\$[\d]+(?:\.\d+)?',   # Simple $XXXXX format
            ]
            
            extracted_number = None
            
            # Try each pattern
            for pattern in numeric_patterns:
                matches = re.findall(pattern, val_str)
                if matches:
                    # Take the longest match (most likely to be the full amount)
                    extracted_number = max(matches, key=len)
                    break
            
            # If no standard pattern found, try to extract any sequence of digits
            if not extracted_number:
                # Look for any sequence of digits (at least 3 digits for meaningful amounts)
                digit_matches = re.findall(r'\d{3,}', val_str)
                if digit_matches:
                    # Take the longest sequence of digits
                    extracted_number = max(digit_matches, key=len)
            
            if not extracted_number:
                print(f"Warning: Could not extract number from '{val_str}'")
                return np.nan
            
            # Clean the extracted number
            cleaned = extracted_number
            
            # Remove currency symbols
            cleaned = re.sub(r'[$€£¥₹₽]', '', cleaned)
            
            # Remove thousands separators (commas)
            cleaned = re.sub(r',', '', cleaned)
            
            # Remove any remaining non-digit characters except decimal points
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            
            # Handle multiple decimal points (keep only the last one)
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            
            # Remove empty strings or just decimal points
            if not cleaned or cleaned == '.' or not re.search(r'\d', cleaned):
                print(f"Warning: No valid number found in '{val_str}' after cleaning")
                return np.nan
            
            return cleaned
        
        return series.apply(clean_currency_value)
    
    def _clean_percentage_column(self, series: pd.Series) -> pd.Series:
        """Clean percentage values"""
        def clean_percentage_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val)
            cleaned = re.sub(r'[^\d.-]', '', val_str)
            
            if not cleaned:
                return np.nan
                
            return cleaned
        
        return series.apply(clean_percentage_value)
    
    def _clean_scientific_column(self, series: pd.Series) -> pd.Series:
        """Clean scientific notation values"""
        def clean_scientific_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val)
            # Scientific notation pattern
            match = re.search(r'[+-]?[0-9]*\.?[0-9]+[eE][+-]?[0-9]+', val_str)
            
            if match:
                return match.group()
            else:
                # Fallback to regular numeric cleaning
                cleaned = re.sub(r'[^\d.-]', '', val_str)
                return cleaned if cleaned else np.nan
        
        return series.apply(clean_scientific_value)
    
    def _clean_generic_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean generic numeric values with improved handling of mixed formats"""
        def clean_generic_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val).strip()
            if not val_str:
                return np.nan
            
            # Handle special cases like "24RK", "4TS3", etc.
            # These appear to be numeric values with suffix codes
            
            # Try to extract the leading numeric part
            numeric_match = re.match(r'^(\d+)', val_str)
            if numeric_match:
                extracted_number = numeric_match.group(1)
                return extracted_number
            
            # If no leading number, try to find any number in the string
            numbers = re.findall(r'\d+(?:\.\d+)?', val_str)
            if numbers:
                # Take the first/longest number found
                return max(numbers, key=len)
            
            # Fallback: remove everything except digits, periods, and minus signs
            cleaned = re.sub(r'[^\d.-]', '', val_str)
            cleaned = re.sub(r',', '', cleaned)  # Remove thousands separators
            
            # Handle multiple periods
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            
            if cleaned and re.search(r'\d', cleaned):
                return cleaned
            else:
                print(f"Warning: Could not extract number from '{val_str}'")
                return np.nan
        
        return series.apply(clean_generic_value)
    
    async def format_dataframe_numerics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Main method to format all numeric fields in a DataFrame using LLM identification"""
        print("🤖 Starting LLM-powered numeric field formatting...")
        
        # Create a copy to avoid modifying original
        formatted_df = df.copy()
        
        # Use LLM to identify numeric columns
        numeric_columns = await self.identify_numeric_columns(formatted_df)
        
        if not numeric_columns:
            print("No numeric columns identified for formatting")
            return formatted_df, {"formatted_columns": [], "errors": []}
        
        print(f"Identified {len(numeric_columns)} numeric columns: {list(numeric_columns.keys())}")
        
        formatting_results = {
            "formatted_columns": [],
            "errors": [],
            "column_info": numeric_columns,
            "identification_method": "llm_gemini"
        }
        
        # Clean each numeric column
        for col_name, numeric_info in numeric_columns.items():
            try:
                print(f"Formatting column: {col_name} (confidence: {numeric_info.get('confidence', 'unknown')})")
                
                # Clean the column
                formatted_df[col_name] = self.clean_numeric_column(formatted_df[col_name], numeric_info)
                
                # Track successful formatting
                formatting_results["formatted_columns"].append({
                    "column": col_name,
                    "type": numeric_info["numeric_type"],
                    "target_dtype": str(formatted_df[col_name].dtype),
                    "confidence": numeric_info.get("confidence", "unknown"),
                    "null_count": formatted_df[col_name].isnull().sum(),
                    "sample_before": df[col_name].head(3).tolist(),
                    "sample_after": formatted_df[col_name].head(3).tolist()
                })
                
                print(f"✅ Successfully formatted {col_name} as {numeric_info['numeric_type']}")
                
            except Exception as e:
                error_msg = f"Failed to format column {col_name}: {str(e)}"
                print(f"✗ {error_msg}")
                formatting_results["errors"].append(error_msg)
        
        return formatted_df, formatting_results

class WebScraper:
    """Handles web scraping functionality"""
    
    async def fetch_webpage(self, url: str) -> str:
        """Fetch webpage content using Playwright with stealth mode"""
        stealth = Stealth()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            context = await browser.new_context()
            await stealth.apply_stealth_async(context)
            
            page = await context.new_page()
            
            # Set additional headers to appear more like a real user
            await page.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            })
            
            try:
                print(f"🌐 Fetching {url} with Playwright stealth mode...")
                await page.goto(url, wait_until='networkidle', timeout=30000)
                content = await page.content()
                await browser.close()
                print("✅ Successfully fetched webpage with Playwright")
                return content
            except Exception as e:
                await browser.close()
                raise Exception(f"Failed to fetch {url}: {str(e)}")
    
    async def extract_table_from_html(self, html_content: str) -> pd.DataFrame:
        """Extract the best table from HTML content using LLM guidance"""
        # First, let LLM analyze the HTML structure and suggest extraction strategy
        extraction_strategy = await self._get_llm_extraction_strategy(html_content)
        
        if extraction_strategy.get("method") == "pandas_direct":
            return await self._pandas_extraction_with_llm_guidance(html_content, extraction_strategy)
        elif extraction_strategy.get("method") == "beautifulsoup_guided":
            return await self._beautifulsoup_extraction_with_llm_guidance(html_content, extraction_strategy)
        else:
            # Fallback to traditional methods
            return await self._fallback_extraction(html_content)
    
    async def _get_llm_extraction_strategy(self, html_content: str) -> Dict[str, Any]:
        """Use LLM to analyze HTML and suggest best extraction strategy"""
        # Get a sample of the HTML (first 8000 chars to avoid token limits)
        html_sample = html_content[:8000]
        
        analysis_prompt = f"""
        Analyze this HTML content and determine the best strategy to extract tabular data:
        
        HTML SAMPLE:
        {html_sample}
        
        Please analyze and return a JSON object with:
        {{
            "method": "pandas_direct" | "beautifulsoup_guided" | "custom_parsing",
            "table_indicators": {{
                "has_html_tables": true/false,
                "table_classes": ["list of CSS classes found on tables"],
                "table_count": number_of_tables_found,
                "best_table_selector": "CSS selector for the main data table",
                "data_structure": "regular_table" | "nested_structure" | "list_based" | "divs_as_table"
            }},
            "extraction_guidance": {{
                "expected_columns": ["list", "of", "expected", "column", "names"],
                "header_location": "first_row" | "th_tags" | "specific_selector",
                "data_row_pattern": "description of how data rows are structured",
                "skip_patterns": ["patterns to skip like navigation rows"],
                "cleaning_needed": ["currency", "references", "special_chars", "multiline"]
            }},
            "pandas_compatibility": {{
                "can_use_pandas": true/false,
                "suggested_params": {{"attrs": {{}}, "skiprows": 0}},
                "reason": "explanation"
            }}
        }}
        
        Focus on finding the MAIN DATA TABLE with the most relevant information, not navigation or sidebar tables.
        Be specific about CSS selectors and patterns you observe.
        """
        
        response = await ping_gemini(
            analysis_prompt, 
            "You are an HTML parsing expert. Analyze the structure and provide specific extraction guidance. Return only valid JSON."
        )
        
        try:
            if "error" in response:
                print(f"❌ LLM analysis failed: {response['error']}")
                return self._fallback_analysis(html_content)
            
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean JSON from markdown if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            strategy = json.loads(response_text)
            print(f"✅ LLM extraction strategy: {strategy.get('method')} for {strategy.get('table_indicators', {}).get('table_count', 0)} tables")
            return strategy
            
        except Exception as e:
            print(f"❌ Error parsing LLM strategy: {e}")
            return self._fallback_analysis(html_content)
    
    def _fallback_analysis(self, html_content: str) -> Dict[str, Any]:
        """Fallback analysis using simple HTML parsing"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        return {
            "method": "beautifulsoup_guided" if tables else "custom_parsing",
            "table_indicators": {
                "has_html_tables": len(tables) > 0,
                "table_classes": list(set([table.get('class', [''])[0] for table in tables if table.get('class')])),
                "table_count": len(tables),
                "best_table_selector": "table",
                "data_structure": "regular_table"
            },
            "extraction_guidance": {
                "expected_columns": [],
                "header_location": "first_row",
                "data_row_pattern": "standard tr/td structure",
                "skip_patterns": [],
                "cleaning_needed": ["references", "special_chars"]
            },
            "pandas_compatibility": {
                "can_use_pandas": len(tables) > 0,
                "suggested_params": {},
                "reason": "Basic table structure detected"
            }
        }
    
    async def _pandas_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use pandas with LLM-guided parameters"""
        print("📊 Using LLM-guided pandas extraction...")
        
        pandas_params = strategy.get("pandas_compatibility", {}).get("suggested_params", {})
        table_indicators = strategy.get("table_indicators", {})
        
        try:
            # Try with LLM-suggested parameters first
            if "attrs" in pandas_params and pandas_params["attrs"]:
                tables = pd.read_html(StringIO(html_content), attrs=pandas_params["attrs"])
            else:
                tables = pd.read_html(StringIO(html_content))
            
            if not tables:
                raise Exception("No tables found with pandas")
            
            # Use LLM guidance to select the best table
            best_table = await self._select_best_table_with_llm(tables, strategy)
            
            # Clean the table using LLM guidance
            cleaned_table = await self._clean_table_with_llm_guidance(best_table, strategy)
            
            print(f"✅ Pandas extraction successful: {cleaned_table.shape}")
            return cleaned_table
            
        except Exception as e:
            print(f"❌ Pandas extraction failed: {e}")
            return await self._beautifulsoup_extraction_with_llm_guidance(html_content, strategy)
    
    async def _select_best_table_with_llm(self, tables: List[pd.DataFrame], strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use LLM to select the best table from multiple candidates"""
        if len(tables) == 1:
            return tables[0]
        
        # Create summary of each table for LLM analysis
        table_summaries = []
        for i, table in enumerate(tables):
            summary = {
                "table_index": i,
                "shape": table.shape,
                "columns": list(table.columns)[:10],  # First 10 columns
                "sample_data": table.head(3).to_dict('records') if not table.empty else [],
                "has_numeric_data": any(table.dtypes == 'object'),  # Look for potential numeric columns
                "null_percentage": round(table.isnull().sum().sum() / (len(table) * len(table.columns)) * 100, 2)
            }
            table_summaries.append(summary)
        
        expected_columns = strategy.get("extraction_guidance", {}).get("expected_columns", [])
        
        selection_prompt = f"""
        I have {len(tables)} tables extracted from a webpage. Help me select the MAIN DATA TABLE with the most relevant information.
        
        EXPECTED DATA: {expected_columns if expected_columns else "General tabular data"}
        
        TABLE SUMMARIES:
        {json.dumps(table_summaries, indent=2, default=str)}
        
        Return a JSON object with:
        {{
            "selected_table_index": 0,  // Index of the best table
            "reason": "explanation of why this table was chosen",
            "confidence": "high" | "medium" | "low"
        }}
        
        Choose the table that:
        1. Has the most relevant data (not navigation/sidebar tables)
        2. Has reasonable size (not too small, not empty)
        3. Has proper structure with meaningful columns
        4. Contains the type of data we're looking for
        """
        
        try:
            response = await ping_gemini(selection_prompt, "You are a data analysis expert. Select the most relevant table. Return only valid JSON.")
            
            if "error" not in response and "candidates" in response:
                response_text = response["candidates"][0]["content"]["parts"][0]["text"]
                
                # Clean JSON
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                selection = json.loads(response_text)
                selected_idx = selection.get("selected_table_index", 0)
                
                if 0 <= selected_idx < len(tables):
                    print(f"✅ LLM selected table {selected_idx}: {selection.get('reason', 'No reason given')}")
                    return tables[selected_idx]
        
        except Exception as e:
            print(f"❌ LLM table selection failed: {e}")
        
        # Fallback: select largest table with most columns
        return max(tables, key=lambda x: len(x) * len(x.columns))
    
    async def _clean_table_with_llm_guidance(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Clean table using LLM guidance"""
        cleaning_needed = strategy.get("extraction_guidance", {}).get("cleaning_needed", [])
        skip_patterns = strategy.get("extraction_guidance", {}).get("skip_patterns", [])
        
        print(f"🧹 Cleaning table with guidance: {cleaning_needed}")
        
        # Basic cleaning
        df_clean = df.copy()
        
        # Remove empty rows and columns
        df_clean = df_clean.dropna(how='all').reset_index(drop=True)
        df_clean = df_clean.loc[:, ~(df_clean.astype(str) == '').all()]
        
        # Apply LLM-guided cleaning
        for clean_type in cleaning_needed:
            if clean_type == "references":
                df_clean = df_clean.applymap(lambda x: re.sub(r'\[\d+\]', '', str(x)) if pd.notna(x) else x)
            elif clean_type == "special_chars":
                df_clean = df_clean.applymap(lambda x: str(x).replace('\xa0', ' ').replace('\u2013', '-') if pd.notna(x) else x)
            elif clean_type == "multiline":
                df_clean = df_clean.applymap(lambda x: ' '.join(str(x).split()) if pd.notna(x) else x)
        
        # Remove header-like rows based on skip patterns
        if skip_patterns or len(df_clean) > 5:
            df_clean = self._remove_duplicate_headers(df_clean)
        
        print(f"✅ Table cleaned: {df.shape} → {df_clean.shape}")
        return df_clean
    
    def _remove_duplicate_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that look like duplicate headers"""
        if len(df) <= 1:
            return df
        
        # Check for rows that match column names
        header_like_indices = []
        column_names_lower = [str(col).lower().strip() for col in df.columns]
        
        for idx, row in df.iterrows():
            row_values_lower = [str(val).lower().strip() for val in row.values]
            
            # Check if row values match column names (fuzzy matching)
            matches = sum(1 for col, val in zip(column_names_lower, row_values_lower) 
                         if col and val and (col in val or val in col))
            
            if matches >= len(df.columns) * 0.6:  # 60% match threshold
                header_like_indices.append(idx)
        
        if header_like_indices:
            print(f"🧹 Removing {len(header_like_indices)} duplicate header rows")
            df = df.drop(header_like_indices).reset_index(drop=True)
        
        return df
    
    async def _beautifulsoup_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use BeautifulSoup with LLM guidance"""
        print("🔄 Using LLM-guided BeautifulSoup extraction...")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        table_indicators = strategy.get("table_indicators", {})
        
        # Find tables using LLM-suggested selector
        selector = table_indicators.get("best_table_selector", "table")
        
        if "." in selector and not selector.startswith("."):
            # Handle class-based selectors
            tables = soup.select(selector)
        else:
            tables = soup.find_all('table')
        
        if not tables:
            raise Exception(f"No tables found with selector: {selector}")
        
        # Score and select best table
        best_table = self._score_and_select_table(tables, strategy)
        
        # Extract data with LLM guidance
        df = await self._extract_table_data_guided(best_table, strategy)
        
        return df
    
    def _score_and_select_table(self, tables, strategy: Dict[str, Any]) -> Any:
        """Score tables and select the best one"""
        best_table = None
        best_score = 0
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:  # Must have header + data
                continue
            
            # Score based on content and structure
            data_cells = sum(len(row.find_all(['td', 'th'])) for row in rows)
            text_content = len(table.get_text(strip=True))
            
            # Prefer tables with more structured content
            score = data_cells * 0.7 + (text_content / 100) * 0.3
            
            if score > best_score:
                best_score = score
                best_table = table
        
        return best_table or tables[0]
    
    async def _extract_table_data_guided(self, table, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Extract table data with LLM guidance"""
        guidance = strategy.get("extraction_guidance", {})
        
        all_rows = table.find_all('tr')
        if not all_rows:
            raise Exception("No rows found in table")
        
        # Extract headers based on guidance
        header_location = guidance.get("header_location", "first_row")
        if header_location == "th_tags":
            # Look for th tags anywhere in the table
            header_cells = table.find_all('th')
            headers = [cell.get_text(strip=True) for cell in header_cells]
        else:
            # Use first row
            header_row = all_rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
        
        # Clean headers
        headers = [self._clean_cell_text(h) for h in headers]
        headers = [h if h else f"Column_{i}" for i, h in enumerate(headers)]
        
        # Extract data rows
        data_rows = []
        for row in all_rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            row_data = [self._clean_cell_text(cell.get_text(strip=True)) for cell in cells]
            
            # Skip empty or irrelevant rows based on guidance
            if not any(cell.strip() for cell in row_data):
                continue
            
            # Ensure row matches header length
            while len(row_data) < len(headers):
                row_data.append('')
            row_data = row_data[:len(headers)]
            
            data_rows.append(row_data)
        
        if not data_rows:
            raise Exception("No data rows extracted")
        
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Apply final cleaning
        return await self._clean_table_with_llm_guidance(df, strategy)
    
    def _clean_cell_text(self, text: str) -> str:
        """Clean individual cell text"""
        if not text:
            return ""
        
        # Remove references [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean whitespace and special characters
        text = text.replace('\xa0', ' ').replace('\u2013', '-').replace('\u2014', '-')
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def _fallback_extraction(self, html_content: str) -> pd.DataFrame:
        """Final fallback extraction method"""
        print("🔄 Using fallback extraction...")
        
        # Try pandas first
        try:
            tables = pd.read_html(StringIO(html_content))
            if tables:
                main_table = max(tables, key=lambda x: len(x) * len(x.columns))
                return self._basic_clean_dataframe(main_table)
        except:
            pass
        
        # Try BeautifulSoup as last resort
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            raise Exception("No extractable data found")
        
        # Use the largest table
        best_table = max(tables, key=lambda t: len(t.find_all('tr')))
        
        # Basic extraction
        rows = best_table.find_all('tr')
        data = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [self._clean_cell_text(cell.get_text()) for cell in cells]
            if any(cell.strip() for cell in row_data):
                data.append(row_data)
        
        if not data:
            raise Exception("No data extracted")
        
        # Create DataFrame with consistent columns
        max_cols = max(len(row) for row in data)
        headers = [f"Column_{i}" for i in range(max_cols)]
        
        # Pad all rows
        for row in data:
            while len(row) < max_cols:
                row.append('')
        
        df = pd.DataFrame(data, columns=headers)
        return self._basic_clean_dataframe(df)
    
    def _basic_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame cleaning"""
        # Remove empty rows and columns
        df = df.dropna(how='all').reset_index(drop=True)
        df = df.loc[:, ~(df.astype(str) == '').all()]
        
        # Remove duplicate headers
        df = self._remove_duplicate_headers(df)
        
        return df
    
    def _beautifulsoup_table_extract(self, html_content: str) -> pd.DataFrame:
        """Extract table using BeautifulSoup with improved parsing"""
        print("🔄 Using BeautifulSoup fallback for table extraction...")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to find wikitable first (Wikipedia standard)
        wikitables = soup.find_all('table', class_='wikitable')
        if wikitables:
            print(f"📊 Found {len(wikitables)} wikitables")
            tables = wikitables
        else:
            # Find all tables and select the largest one by number of rows
            tables = soup.find_all('table')
            if not tables:
                raise Exception("No tables found in HTML")
            print(f"📊 Found {len(tables)} total tables")
        
        # Get the table with most meaningful rows (data)
        best_table = None
        max_score = 0
        
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) < 3:  # Skip very small tables
                continue
                
            # Score table based on size and structure
            data_rows = 0
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:  # Must have at least 2 columns
                    cell_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                    if len(cell_text.strip()) > 10:  # Must have meaningful content
                        data_rows += 1
            
            score = data_rows * len(rows[0].find_all(['td', 'th']) if rows else [])
            
            if score > max_score:
                max_score = score
                best_table = table
                print(f"📊 Table {i+1}: {len(rows)} rows, score {score} (current best)")
        
        if not best_table:
            raise Exception("No suitable table found")
        
        return self._extract_table_data(best_table)
    
    def _extract_table_data(self, table) -> pd.DataFrame:
        """Extract clean data from a BeautifulSoup table object"""
        all_rows = table.find_all('tr')
        
        # Extract headers from first row
        header_row = all_rows[0]
        headers = []
        for cell in header_row.find_all(['th', 'td']):
            header_text = cell.get_text(strip=True)
            # Clean header text
            header_text = re.sub(r'\[\d+\]', '', header_text)  # Remove reference numbers
            header_text = ' '.join(header_text.split())  # Clean whitespace
            headers.append(header_text if header_text else f"Column_{len(headers)}")
        
        print(f"📊 Extracted headers: {headers}")
        
        # Extract data rows (skip header)
        data_rows = []
        for row_idx, row in enumerate(all_rows[1:], 1):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
                
            row_data = []
            for cell in cells:
                # Get clean cell text
                cell_text = cell.get_text(strip=True)
                # Remove reference numbers and clean
                cell_text = re.sub(r'\[\d+\]', '', cell_text)
                cell_text = ' '.join(cell_text.split())
                
                # Handle special characters and encoding issues
                cell_text = cell_text.replace('\u2013', '-').replace('\u2014', '-')  # em/en dashes
                cell_text = cell_text.replace('\xa0', ' ')  # non-breaking space
                
                row_data.append(cell_text)
            
            # Only add rows with meaningful data
            if any(cell.strip() for cell in row_data):
                # Pad row to match header length
                while len(row_data) < len(headers):
                    row_data.append('')
                # Trim row if it's too long
                row_data = row_data[:len(headers)]
                data_rows.append(row_data)
        
        # Create DataFrame
        if not data_rows:
            raise Exception("No data rows found in table")
        
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Clean the dataframe
        df = self._post_process_dataframe(df)
        
        print(f"📊 Final DataFrame shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        return df
    
    def _post_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the dataframe to clean up common issues"""
        # Remove rows that are duplicates of headers
        if len(df) > 1:
            # Check if any row contains header-like values
            header_like_rows = []
            for idx, row in df.iterrows():
                header_similarity = sum(1 for col in df.columns 
                                      if str(row[col]).strip().lower() == col.strip().lower())
                if header_similarity > len(df.columns) * 0.6:  # 60% similarity threshold
                    header_like_rows.append(idx)
            
            if header_like_rows:
                print(f"📊 Removing {len(header_like_rows)} header-like rows")
                df = df.drop(header_like_rows).reset_index(drop=True)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').reset_index(drop=True)  # Remove empty rows
        df = df.loc[:, ~(df == '').all()]  # Remove empty columns
        
        # Remove rows with mostly empty cells
        threshold = max(1, len(df.columns) // 3)  # At least 1/3 of columns should have data
        df = df.dropna(thresh=threshold).reset_index(drop=True)
        
        return df

class ImprovedWebScraper:
    """Main class that coordinates web scraping and numeric formatting"""
    
    def __init__(self):
        self.numeric_formatter = NumericFieldFormatter()
        self.web_scraper = WebScraper()
    
    async def extract_data(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to extract data from web sources"""
        # Handle both URL string and config dict formats
        if isinstance(source_config, str):
            url = source_config
        else:
            url = source_config.get("url", "")
            if not url:
                raise Exception("No URL provided in source config")
        
        print(f"🚀 Starting data extraction for: {url}")
        
        # Fetch webpage
        html_content = await self.web_scraper.fetch_webpage(url)
        
        # Extract table data
        df = await self.web_scraper.extract_table_from_html(html_content)
        
        if df.empty:
            raise Exception(f"No data extracted from {url}")
        
        print(f"📊 Raw data extracted: {df.shape}")
        
        # Clean numeric fields using LLM
        cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(df)
        
        print(f"✅ Data cleaning complete: {cleaned_df.shape}")
        
        return {
            "dataframe": cleaned_df,
            "metadata": {
                "source_type": "web_scrape",
                "source_url": url,
                "extraction_method": "automated_table_detection",
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "data_types": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
                "numeric_formatting": formatting_results
            }
        }
    
    async def scrape_and_clean(self, url: str) -> Dict[str, Any]:
        """Alias method for backward compatibility"""
        return await self.extract_data(url)

# Example usage:
