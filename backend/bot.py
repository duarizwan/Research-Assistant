
__all__ = ['search_papers', 'summarize_with_gemini', 'download_papers_enhanced', 'setup_logging', 'safe_filename']

import arxiv
import json
import os
import time
import re
import unicodedata
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

PAPER_DIR = "papers"
MAX_CONCURRENT_DOWNLOADS = 3
GEMINI_RETRY_ATTEMPTS = 3
GEMINI_RATE_LIMIT_DELAY = 2  # seconds between requests
METADATA_FILE = "papers_metadata.json"

# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler
    file_handler = logging.FileHandler(f'logs/research_assistant_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Configure logger
    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
    return logging.getLogger(__name__)

# Initialize logger (will be reconfigured in main)
logger = logging.getLogger(__name__)

# -------------------------------
# Configuration & Setup
# -------------------------------
def setup_gemini():
    """Setup Gemini API with proper error handling."""
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        logger.error("GENAI_API_KEY not found in environment variables")
        raise SystemExit("❌ Error: Please set GENAI_API_KEY in your .env file")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini API configured successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
        raise SystemExit(f"❌ Error configuring Gemini: {e}")

# Initialize Gemini model
model = setup_gemini()

# -------------------------------
# Utility Functions
# -------------------------------
def safe_filename(title: str, max_length: int = 80) -> str:
    """Create a safe filename from paper title with comprehensive sanitization."""
    if not title:
        return "untitled"
    
    # Normalize unicode characters
    title = unicodedata.normalize("NFKD", title)
    
    # Remove or replace problematic characters
    title = re.sub(r'[<>:"/\\|?*]', '_', title)  # Windows forbidden chars
    title = re.sub(r'[^\w\-_\. ]', '_', title)   # Keep only safe chars
    title = re.sub(r'_+', '_', title)            # Collapse multiple underscores
    title = re.sub(r'\s+', '_', title)           # Replace spaces with underscores
    
    # Truncate and clean up
    title = title[:max_length].strip('_.')
    
    # Ensure not empty and not reserved names
    if not title or title.lower() in ['con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9']:
        title = "paper"
    
    return title

def validate_input_field(field: str) -> bool:
    """Validate research field input with improved criteria."""
    field = field.strip()
    if len(field) < 2:
        return False
    if not any(c.isalpha() for c in field):
        return False
    # Check for reasonable length (not just random characters)
    if len(field) > 100:
        return False
    return True

def normalize_date(date_str: str) -> str:
    """Normalize date string to ISO YYYY-MM-DD format."""
    try:
        if isinstance(date_str, str) and len(date_str) == 10 and date_str.count('-') == 2:
            return date_str  # Already in correct format
        # Handle other date formats if needed
        return str(date_str)
    except:
        return "unknown"

def get_valid_field_input() -> str:
    """Get valid field input from user with validation."""
    while True:
        field = input("👉 Enter a research field (e.g., Computer Science, Physics, Biology): ").strip()
        if field.lower() in ['quit', 'exit', 'q']:
            return field
        
        if validate_input_field(field):
            logger.debug(f"Valid field input received: {field}")
            return field
        else:
            print("⚠️  Please enter a valid research field (2-100 characters with letters).")
            print("   Examples: Computer Science, Machine Learning, Quantum Physics")

def get_valid_topic_input(field: str) -> str:
    """Get valid topic input from user with validation."""
    while True:
        topic = input(f"👉 Enter a topic within '{field}' (or 'help' for suggestions): ").strip()
        if topic.lower() in ['quit', 'exit', 'q']:
            return topic
        
        if topic.lower() == 'help':
            show_topic_suggestions(field)
            continue
            
        if validate_input_field(topic):
            logger.debug(f"Valid topic input received: {topic}")
            return topic
        else:
            print("⚠️  Please enter a valid topic (2-100 characters with letters).")
            print("   Examples: neural networks, deep learning, natural language processing")

def show_topic_suggestions(field: str):
    """Show topic suggestions based on field."""
    suggestions = {
        "computer science": ["machine learning", "artificial intelligence", "neural networks", "deep learning", "computer vision", "natural language processing"],
        "physics": ["quantum mechanics", "relativity", "particle physics", "condensed matter", "astrophysics", "thermodynamics"],
        "biology": ["genetics", "evolution", "molecular biology", "cell biology", "neuroscience", "bioinformatics"],
        "mathematics": ["algebra", "calculus", "statistics", "topology", "number theory", "optimization"],
        "chemistry": ["organic chemistry", "inorganic chemistry", "physical chemistry", "biochemistry", "materials science"]
    }
    
    field_lower = field.lower()
    for key, topics in suggestions.items():
        if key in field_lower or any(word in field_lower for word in key.split()):
            print(f"💡 Suggested topics for {field}: {', '.join(topics)}")
            return
    
    print("💡 Try specific topics like: deep learning, quantum computing, gene therapy, etc.")

# -------------------------------
# Metadata Management
# -------------------------------
def load_metadata() -> dict:
    """Load existing metadata from file."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Loaded metadata for {len(data)} papers")
                return data
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
    return {}

def save_metadata(metadata: dict):
    """Save metadata to file."""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata for {len(metadata)} papers")
    except Exception as e:
        logger.error(f"Could not save metadata: {e}")

# -------------------------------
# ArXiv Paper Search with Pagination
# -------------------------------
def search_papers(query: str, max_results: int = 10, start: int = 0) -> Tuple[Dict[str, dict], bool]:
    """
    Search for papers on arXiv with pagination support.
    Returns (results_dict, has_more_results)
    """
    try:
        logger.info(f"Searching arXiv: '{query}' (results {start+1}-{start+max_results})")
        client = arxiv.Client()
        
        # Search for a few extra to check if more results exist
        search = arxiv.Search(
            query=query,
            max_results=max_results + 5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = list(client.results(search))
        
        # Take only the requested slice
        paper_slice = papers[start:start + max_results] if start < len(papers) else []
        has_more = len(papers) > start + max_results
        
        results = {}
        for paper in paper_slice:
            pid = paper.get_short_id()
            info = {
                "title": paper.title.strip(),
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary.strip(),
                "pdf_url": paper.pdf_url,
                "published": normalize_date(str(paper.published.date())),
                "version": pid.split("v")[-1] if "v" in pid else "1",
                "status": "archived" if paper.journal_ref else "preprint",
                "categories": [cat for cat in paper.categories],
                "arxiv_id": pid
            }
            results[pid] = info
        
        logger.info(f"Found {len(results)} papers (more available: {has_more})")
        return results, has_more
        
    except Exception as e:
        logger.error(f"Error searching arXiv: {e}")
        print(f"⚠️  Error searching arXiv: {e}")
        return {}, False

# -------------------------------
# Improved Selection Logic
# -------------------------------
def parse_year_selection(year_str: str, paper_list: List[tuple]) -> List[str]:
    """Parse year selection with strict regex matching."""
    if not re.match(r'^(19|20)\d{2}$', year_str):
        return []
    
    matches = []
    for pid, info in paper_list:
        if year_str in info["published"]:
            matches.append(pid)
    return matches

def parse_author_selection(author_name: str, paper_list: List[tuple]) -> List[str]:
    """Parse author selection with improved matching."""
    author_name_lower = author_name.lower().strip()
    matches = []
    
    for pid, info in paper_list:
        for author in info["authors"]:
            author_lower = author.lower()
            # Check for exact word match or last name match
            if (author_name_lower in author_lower or 
                any(author_name_lower == part for part in author_lower.split()) or
                author_lower.split()[-1] == author_name_lower):  # Last name match
                matches.append(pid)
                break
    return matches

def parse_selection_improved(selection_input: List[str], paper_list: List[tuple], num_requested: int) -> List[str]:
    """Improved selection parsing with better matching and preview."""
    chosen_papers = []
    
    for sel in selection_input:
        sel = sel.strip()
        if not sel:
            continue
        
        logger.debug(f"Processing selection: {sel}")
        
        if sel.isdigit():  # Serial number selection
            idx = int(sel)
            if 1 <= idx <= len(paper_list):
                pid = paper_list[idx - 1][0]
                if pid not in chosen_papers:
                    chosen_papers.append(pid)
                    logger.debug(f"Added paper by index {idx}: {pid}")
        elif re.match(r'^(19|20)\d{2}$', sel):  # Year selection (strict)
            year_matches = parse_year_selection(sel, paper_list)
            logger.debug(f"Year {sel} matched {len(year_matches)} papers")
            
            if len(year_matches) > num_requested:
                print(f"🔍 Found {len(year_matches)} papers from {sel}:")
                for i, pid in enumerate(year_matches, 1):
                    info = next(info for p, info in paper_list if p == pid)
                    print(f"   {i:2}) {info['title'][:65]}... ({pid})")
                
                indices_input = input(f"Select which papers by indices (1-{len(year_matches)}, comma-separated): ")
                try:
                    indices = [int(x.strip()) for x in indices_input.split(',') if x.strip().isdigit()]
                    for idx in indices:
                        if 1 <= idx <= len(year_matches):
                            pid = year_matches[idx-1]
                            if pid not in chosen_papers:
                                chosen_papers.append(pid)
                except ValueError:
                    print("⚠️  Invalid indices format")
            else:
                chosen_papers.extend([pid for pid in year_matches if pid not in chosen_papers])
                
        else:  # Author name selection
            author_matches = parse_author_selection(sel, paper_list)
            logger.debug(f"Author '{sel}' matched {len(author_matches)} papers")
            
            if len(author_matches) > num_requested:
                print(f"🔍 Found {len(author_matches)} papers by authors matching '{sel}':")
                for i, pid in enumerate(author_matches, 1):
                    info = next(info for p, info in paper_list if p == pid)
                    matching_authors = [a for a in info['authors'] if sel.lower() in a.lower()]
                    print(f"   {i:2}) {info['title'][:50]}... by {', '.join(matching_authors)} ({pid})")
                
                indices_input = input(f"Select which papers by indices (1-{len(author_matches)}, comma-separated): ")
                try:
                    indices = [int(x.strip()) for x in indices_input.split(',') if x.strip().isdigit()]
                    for idx in indices:
                        if 1 <= idx <= len(author_matches):
                            pid = author_matches[idx-1]
                            if pid not in chosen_papers:
                                chosen_papers.append(pid)
                except ValueError:
                    print("⚠️  Invalid indices format")
            else:
                chosen_papers.extend([pid for pid in author_matches if pid not in chosen_papers])
    
    return chosen_papers[:num_requested]  # Limit to requested number

# -------------------------------
# Gemini Integration with Better Error Handling
# -------------------------------
def summarize_with_gemini(paper_info: dict) -> str:
    """Ask Gemini to provide a short summary for a paper with improved retry logic."""
    for attempt in range(GEMINI_RETRY_ATTEMPTS):
        try:
            if attempt > 0:
                logger.debug(f"Gemini retry attempt {attempt + 1} for paper {paper_info.get('title', 'Unknown')[:30]}...")
            
            time.sleep(GEMINI_RATE_LIMIT_DELAY)  # Rate limiting
            
            prompt = f"""Summarize this research paper briefly in 2-3 sentences focusing on the main contribution and methodology:

Title: {paper_info['title']}
Authors: {", ".join(paper_info['authors'][:5])}
Published: {paper_info['published']}
Categories: {", ".join(paper_info.get('categories', []))}
Abstract: {paper_info['summary'][:800]}...

Provide a concise summary highlighting the key innovation and practical implications."""

            response = model.generate_content(prompt)
            summary = response.text.strip()
            logger.debug(f"Successfully generated summary for {paper_info.get('title', 'Unknown')[:30]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt < GEMINI_RETRY_ATTEMPTS - 1:
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                logger.error(f"All Gemini attempts failed for paper {paper_info.get('title', 'Unknown')}")
                # Enhanced fallback
                abstract_preview = paper_info['summary'][:300]
                return f"📄 Summary unavailable (Gemini service error). Abstract preview: {abstract_preview}{'...' if len(paper_info['summary']) > 300 else ''}"

# -------------------------------
# Enhanced Download System with Resume Support
# -------------------------------
def get_file_size(url: str) -> Optional[int]:
    """Get file size from URL headers."""
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            return int(response.headers.get('Content-Length', 0))
    except:
        pass
    return None

def download_single_paper_enhanced(pid: str, paper_info: dict, download_path: str, resume: bool = True, web_mode: bool = False) -> tuple:
    """Download a single paper with resume capability and better error handling."""
    pdf_url = paper_info.get('pdf_url') or f"https://arxiv.org/pdf/{pid}.pdf"
    safe_title = safe_filename(paper_info['title'])
    pdf_filename = f"{pid}_{safe_title}.pdf"
    pdf_path = os.path.join(download_path, pdf_filename)
    temp_path = pdf_path + ".part"
    
    logger.info(f"Starting download: {pid}")
    
    # Check if file exists and handle appropriately
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path)
        if file_size > 1024:  # More than 1KB, likely complete
            if web_mode:
                # In web mode, skip existing files
                logger.info(f"Skipped existing file: {pid}")
                return pid, "skipped", pdf_path
            else:
                choice = input(f"📄 File {pdf_filename} already exists ({file_size//1024}KB). (o)verwrite, (s)kip, or (r)ename? ").lower()
                if choice == 's':
                    logger.info(f"Skipped existing file: {pid}")
                    return pid, "skipped", pdf_path
                elif choice == 'r':
                    counter = 1
                    base_name = f"{pid}_{safe_title}"
                    while os.path.exists(pdf_path):
                        pdf_filename = f"{base_name}_({counter}).pdf"
                        pdf_path = os.path.join(download_path, pdf_filename)
                        temp_path = pdf_path + ".part"
                        counter += 1
    
    # Handle resumable download
    resume_header = {}
    start_pos = 0
    if resume and os.path.exists(temp_path):
        start_pos = os.path.getsize(temp_path)
        resume_header = {'Range': f'bytes={start_pos}-'}
        logger.debug(f"Resuming download from byte {start_pos}")
        if not web_mode:
            print(f"⏯️  Resuming download of {pid} from {start_pos} bytes...")
    else:
        if not web_mode:
            print(f"⬇️  Downloading {pid}...")
    
    try:
        response = requests.get(pdf_url, stream=True, headers=resume_header, timeout=30)
        
        if response.status_code in [200, 206]:  # 206 for partial content
            mode = "ab" if start_pos > 0 else "wb"
            with open(temp_path, mode) as f:
                downloaded = start_pos
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress indicator
                        if downloaded % (1024 * 100) == 0 and not web_mode:  # Every 100KB
                            print(f"   📥 {downloaded // 1024}KB downloaded...")
            
            # Verify download and rename
            final_size = os.path.getsize(temp_path)
            if final_size > 1024:  # At least 1KB
                os.replace(temp_path, pdf_path)
                logger.info(f"Successfully downloaded {pid} ({final_size//1024}KB)")
                return pid, "success", pdf_path
            else:
                os.remove(temp_path)
                logger.error(f"Downloaded file too small for {pid}: {final_size} bytes")
                return pid, "file_too_small", None
        else:
            logger.error(f"HTTP error {response.status_code} for {pid}")
            return pid, f"HTTP_{response.status_code}", None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading {pid}: {e}")
        return pid, f"network_error: {str(e)}", None
    except Exception as e:
        logger.error(f"Unexpected error downloading {pid}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return pid, f"error: {str(e)}", None

def download_papers_enhanced(paper_ids: List[str], topic: str, field: str, papers_info: dict, preview_mode: bool = False) -> List[tuple]:
    """Enhanced paper download with preview mode and better organization."""
    path = os.path.join(PAPER_DIR, field.lower().replace(" ", "_"), topic.lower().replace(" ", "_"))
    
    if preview_mode:
        print(f"\n🔍 PREVIEW MODE - Files would be downloaded to: {path}")
        for pid in paper_ids:
            if pid in papers_info:
                info = papers_info[pid]
                safe_title = safe_filename(info['title'])
                filename = f"{pid}_{safe_title}.pdf"
                print(f"   📄 {filename}")
        return []
    
    os.makedirs(path, exist_ok=True)
    logger.info(f"Downloading {len(paper_ids)} papers to {path}")
    
    successful_downloads = []
    failed_downloads = []
    skipped_downloads = []
    
    # Load and update metadata
    metadata = load_metadata()
    
    # Use ThreadPoolExecutor for controlled concurrency
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        # Submit download tasks
        future_to_pid = {
            executor.submit(download_single_paper_enhanced, pid, papers_info[pid], path): pid 
            for pid in paper_ids if pid in papers_info
        }
        
        # Process completed downloads
        for future in as_completed(future_to_pid):
            pid, status, file_path = future.result()
            
            # Update metadata
            if pid in papers_info:
                metadata[pid] = papers_info[pid].copy()
                metadata[pid]['download_status'] = status
                metadata[pid]['download_path'] = file_path
                metadata[pid]['download_timestamp'] = datetime.now().isoformat()
            
            if status == "success":
                successful_downloads.append((pid, file_path))
                print(f"✅ Successfully downloaded: {pid}")
            elif status == "skipped":
                skipped_downloads.append(pid)
                print(f"⏭️  Skipped existing: {pid}")
            else:
                failed_downloads.append((pid, status))
                print(f"❌ Failed to download {pid}: {status}")
    
    # Save updated metadata
    save_metadata(metadata)
    
    # Print summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {len(successful_downloads)}")
    print(f"   ⏭️  Skipped: {len(skipped_downloads)}")
    print(f"   ❌ Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"   📋 Failed papers: {[pid for pid, _ in failed_downloads]}")
        logger.info(f"Failed downloads: {failed_downloads}")
    
    return successful_downloads

# -------------------------------
# Enhanced Chat Loop
# -------------------------------
def chat_loop(verbose: bool = False, preview_mode: bool = False):
    """Enhanced chat loop with pagination and improved UX."""
    print("🤖 Welcome to the Enhanced Research Assistant! Type 'quit' to exit.\n")
    if verbose:
        print("🔍 Verbose mode enabled - detailed logging active")
    if preview_mode:
        print("👁️  Preview mode enabled - no actual downloads will occur")
    print("💡 Make sure you have GENAI_API_KEY set in your .env file.\n")

    while True:
        print("🔍 Let's find some research papers.")
        
        # Get validated inputs
        field = get_valid_field_input()
        if field.lower() in ['quit', 'exit', 'q']:
            break

        topic = get_valid_topic_input(field)
        if topic.lower() in ['quit', 'exit', 'q']:
            break

        query = f"{field} {topic}"
        print(f"🔍 Searching for: {query}")
        
        # Search with pagination
        start_index = 0
        all_papers = {}
        
        while True:
            papers_batch, has_more = search_papers(query, max_results=10, start=start_index)
            
            if not papers_batch and start_index == 0:
                print("⚠️  No papers found. Try different keywords or check your internet connection.")
                break
                
            all_papers.update(papers_batch)
            
            # Show current batch
            found_count = len(papers_batch)
            total_count = len(all_papers)
            print(f"\n📄 Found {found_count} papers (showing {start_index+1}-{start_index+found_count}, total loaded: {total_count}):\n")
            
            # Show list
            paper_list = list(papers_batch.items())
            for idx, (pid, info) in enumerate(paper_list, start=start_index+1):
                authors_display = ', '.join(info['authors'][:2])
                if len(info['authors']) > 2:
                    authors_display += f" et al. ({len(info['authors'])} authors)"
                
                print(f"{idx:2}) [{info['published']}] {info['title'][:70]}...")
                print(f"    📝 {authors_display}")
                print(f"    🏷️  {', '.join(info.get('categories', [])[:2])} (ID: {pid})")
                print()

            # Pagination options
            if has_more:
                more_choice = input(f"📄 Load more results? (y/n/search): ").lower()
                if more_choice == 'y':
                    start_index += 10
                    continue
                elif more_choice == 'search':
                    break
            
            # Selection process
            paper_list = list(all_papers.items())
            total_available = len(all_papers)
            
            if total_available == 0:
                continue
                
            # Get number to download
            while True:
                try:
                    num_input = input(f"\n👉 How many papers to download? (1-{min(total_available, 20)}): ").strip()
                    if num_input.lower() in ['quit', 'exit', 'q']:
                        return
                    
                    num_to_download = int(num_input)
                    if 1 <= num_to_download <= min(total_available, 20):
                        break
                    else:
                        print(f"⚠️  Please enter a number between 1 and {min(total_available, 20)}")
                except ValueError:
                    print("⚠️  Please enter a valid number")

            print("\n👉 Selection options:")
            print("   - Serial numbers: 1,3,5")
            print("   - Author name: Goodfellow")  
            print("   - Year: 2020")
            print("   - Mixed: 1,Goodfellow,2020")
            print("   - 'help' for more guidance\n")

            selection_input = input("Enter your selection: ").strip()
            if selection_input.lower() in ['quit', 'exit', 'q']:
                return
            if selection_input.lower() == 'help':
                print("\n💡 Selection Help:")
                print("   - Use paper numbers from the list above")
                print("   - Type author's last name (e.g., 'Smith' matches 'John Smith')")
                print("   - Use 4-digit years (e.g., '2020', '2023')")
                print("   - Combine methods with commas: '1,2,Smith,2020'")
                continue
                
            selection_list = [s.strip() for s in selection_input.split(",") if s.strip()]
            
            if not selection_list:
                print("⚠️  No valid selection provided.")
                continue

            chosen_papers = parse_selection_improved(selection_list, paper_list, num_to_download)

            # Handle selection results
            if len(chosen_papers) == 0:
                print("⚠️  No matching papers found for your selection.")
                continue
            elif len(chosen_papers) < num_to_download:
                print(f"⚠️  You requested {num_to_download} papers but selected {len(chosen_papers)}.")
                choice = input(f"Continue with {len(chosen_papers)} papers? (y/n): ").lower()
                if choice != 'y':
                    continue

            # Show final selection preview
            print(f"\n📋 Final selection ({len(chosen_papers)} papers):")
            for i, pid in enumerate(chosen_papers, 1):
                info = all_papers[pid]
                print(f"{i:2}) {info['title'][:65]}... ({pid})")
            
            # Preview or download confirmation
            if preview_mode:
                download_papers_enhanced(chosen_papers, topic, field, all_papers, preview_mode=True)
            else:
                confirm = input(f"\n✅ Proceed with download? (y/n/preview): ").lower()
                if confirm == 'preview':
                    download_papers_enhanced(chosen_papers, topic, field, all_papers, preview_mode=True)
                    confirm = input(f"Now proceed with actual download? (y/n): ").lower()
                
                if confirm != 'y':
                    continue

                # Download papers
                print(f"\n⬇️  Starting download of {len(chosen_papers)} papers...")
                successful_downloads = download_papers_enhanced(chosen_papers, topic, field, all_papers)

                # Show paper details and summaries
                if successful_downloads:
                    print(f"\n📑 Paper Details and AI-Generated Summaries:")
                    print("=" * 80)
                    
                    for pid, file_path in successful_downloads:
                        if pid in all_papers:
                            info = all_papers[pid]
                            print(f"\n🔹 {info['title']}")
                            print(f"📅 Published: {info['published']} | Version: v{info['version']} | Status: {info['status']}")
                            print(f"👥 Authors: {', '.join(info['authors'])}")
                            print(f"🏷️  Categories: {', '.join(info.get('categories', []))}")
                            print(f"📄 Local file: {os.path.basename(file_path)}")
                            print(f"🔗 ArXiv URL: {info['pdf_url']}")
                            
                            print("🤖 AI Summary: ", end="")
                            summary = summarize_with_gemini(info)
                            print(summary)
                            print("-" * 80)

            break  # Exit pagination loop

        # Ask for continuation
        if all_papers:  # Only ask if we had results
            continue_choice = input(f"\n🔄 Search for more papers? (y/n): ").lower()
            if continue_choice != 'y':
                break

    print("👋 Thank you for using the Enhanced Research Assistant!")


def show_help():
    """Show comprehensive help information."""
    help_text = """
🤖 Enhanced Research Assistant - Help

FEATURES:
- Smart paper search with pagination
- Multiple selection methods (index, author, year)  
- Resume interrupted downloads
- AI-powered summaries via Gemini
- Comprehensive logging and error handling
- Preview mode for testing selections

USAGE:
1. Enter research field (e.g., "Computer Science", "Physics")
2. Enter specific topic (e.g., "machine learning", "quantum computing")
3. Browse results with pagination (load more if needed)
4. Select papers by:
   - Numbers: 1,3,5 (from the displayed list)
   - Authors: Smith,Goodfellow (matches author names)
   - Years: 2020,2023 (matches publication years)
   - Mixed: 1,Smith,2020 (combine methods)

SELECTION TIPS:
- Author matching is flexible (matches partial names)
- Year must be 4 digits (1900-2099)
- Large author/year matches show preview for selection
- Use 'help' during selection for more guidance

COMMANDS:
- 'quit'/'exit'/'q' - Exit at any prompt
- 'help' - Show topic suggestions or selection help
- 'preview' - Preview downloads without actually downloading

FILES & ORGANIZATION:
- Papers saved to: papers/field_name/topic_name/
- Metadata tracked in: papers_metadata.json
- Logs saved to: logs/research_assistant_TIMESTAMP.log
- Safe filename generation (handles special characters)

ENVIRONMENT:
- Requires GENAI_API_KEY in .env file
- Supports resume of interrupted downloads
- Rate limiting for API calls
- Concurrent downloads (max 3 simultaneous)
"""
    print(help_text)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Research Assistant for arXiv paper discovery and download",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose logging and debug output'
    )
    
    parser.add_argument(
        '-p', '--preview', 
        action='store_true',
        help='Enable preview mode (no actual downloads)'
    )
    
    parser.add_argument(
        '--help-extended',
        action='store_true',
        help='Show extended help and usage examples'
    )
    
    args = parser.parse_args()
    
    if args.help_extended:
        show_help()
        return
    
    # Setup logging with verbosity
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        logger.info("Starting Enhanced Research Assistant")
        chat_loop(verbose=args.verbose, preview_mode=args.preview)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using the Dua's Research Bot!")
        logger.info("Application terminated by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print("Please check the logs and your configuration.")


if __name__ == "__main__":
    main()