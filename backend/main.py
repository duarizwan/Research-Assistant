
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import asyncio
import logging
from datetime import datetime
import os
import re
import shutil

# Import your existing research assistant functions
from bot import (
    search_papers, 
    summarize_with_gemini, 
    download_papers_enhanced,
    setup_logging,
    safe_filename
)

app = FastAPI(title="Research Assistant API", version="1.0.0")

# Setup logging
logger = setup_logging(verbose=True)

# Enable CORS for Next.js frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        FRONTEND_URL,
        "https://*.vercel.app"  # Allow Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (downloaded papers) if papers directory exists
if os.path.exists("papers"):
    app.mount("/papers", StaticFiles(directory="papers"), name="papers")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    workflow_step: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    field: Optional[str] = None
    topic: Optional[str] = None
    max_results: Optional[int] = 10

class Paper(BaseModel):
    id: str
    title: str
    authors: List[str]
    summary: str
    pdf_url: str
    published: str
    version: str
    status: str
    categories: List[str] = []

class ChatResponse(BaseModel):
    response: str
    papers: Optional[List[Paper]] = None
    action_type: str = "search"
    session_id: str = "default"

class ConversationState:
    def __init__(self):
        self.current_papers = {}
        self.current_field = ""
        self.current_topic = ""
        self.awaiting_selection = False
        self.awaiting_field = False
        self.awaiting_topic = False
        self.conversation_history = []
        self.global_paper_map = {}  # Maps sequential numbers to paper IDs
        self.next_serial_number = 1  # Tracks the next available serial number

# Session management (in production, use Redis or similar)
sessions: Dict[str, ConversationState] = {}

def get_session(session_id: str) -> ConversationState:
    if session_id not in sessions:
        sessions[session_id] = ConversationState()
        print(f"🆕 NEW SESSION CREATED: {session_id} with next_serial_number=1")
        logger.info(f"Created new session for session_id: {session_id} with fresh numbering (next_serial_number=1)")
    else:
        print(f"♻️ EXISTING SESSION USED: {session_id} with next_serial_number={sessions[session_id].next_serial_number}")
    return sessions[session_id]

def get_project_papers_directory() -> str:
    """Get the project's papers directory path"""
    try:
        # Get the project root directory (parent of backend)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        papers_dir = os.path.join(project_root, "papers")
        return papers_dir
    except Exception as e:
        logger.error(f"Error getting project papers directory: {e}")
        return "papers"

@app.get("/")
async def root():
    return {"message": "Research Assistant API is running"}

@app.post("/api/reset")
async def reset_session(session_id: str = "default"):
    """Reset session state to ensure clean start"""
    try:
        if session_id in sessions:
            del sessions[session_id]
        logger.info(f"Session reset for session_id: {session_id}")
        return {"message": "Session reset successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return {"error": "Failed to reset session"}

@app.get("/api/reset")
async def reset_session_get(session_id: str = "default"):
    """Reset session state via GET request (for page refresh detection)"""
    try:
        print(f"🔄 RESET ENDPOINT CALLED: session_id={session_id}")
        print(f"📊 Current sessions: {list(sessions.keys())}")
        
        session_existed = session_id in sessions
        if session_existed:
            del sessions[session_id]
            print(f"✅ Session deleted: {session_id}")
            logger.info(f"Session reset via GET for session_id: {session_id} (session existed)")
        else:
            print(f"ℹ️ No existing session to delete: {session_id}")
            logger.info(f"Session reset via GET for session_id: {session_id} (no existing session)")
        
        print(f"📊 Sessions after reset: {list(sessions.keys())}")
        return {"message": "Session reset successfully", "session_id": session_id, "session_existed": session_existed}
    except Exception as e:
        print(f"❌ Error in reset endpoint: {e}")
        logger.error(f"Error resetting session via GET: {e}")
        return {"error": "Failed to reset session"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, background_tasks: BackgroundTasks):
    try:
        session = get_session(message.session_id)
        logger.info(f"Received message: {message.message}, workflow_step: {message.workflow_step}")
        
        # Handle restart commands (q, quit, restart)
        user_input = message.message.lower().strip()
        if user_input in ['q', 'quit', 'restart', 'start over', 'new search']:
            # Reset session state
            session.__init__()
            logger.info(f"Session reset for session_id: {message.session_id}")
            return ChatResponse(
                response="🔄 Starting fresh!\n\nWhich field of research are you looking for papers in? (e.g., Computer Science, Biology, Physics, Mathematics, Chemistry)",
                action_type="restart",
                session_id=message.session_id
            )
        
        # Handle different workflow steps
        if message.workflow_step == "search":
            max_results = message.max_results or 10
            response = await handle_paper_search(message.message, session, max_results)
        elif message.workflow_step == "load_more":
            max_results = message.max_results or 5
            response = await handle_load_more_papers(message.message, session, max_results)
        elif message.workflow_step == "download":
            response = await handle_download_request(message, session, background_tasks)
        else:
            # Legacy support for old workflow
            if session.awaiting_field:
                response = await handle_field_input(message.message, session)
            elif session.awaiting_topic:
                response = await handle_topic_input(message.message, session)
            elif session.awaiting_selection:
                response = await handle_selection_input(message.message, session, background_tasks)
            else:
                response = await handle_initial_query(message.message, session)
        
        response.session_id = message.session_id
        return response
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response="I encountered an error processing your request. Please try again or start a new search.",
            action_type="error",
            session_id=message.session_id
        )

async def handle_initial_query(user_message: str, session: ConversationState) -> ChatResponse:
    """Handle the initial query to determine field and topic"""
    user_input = user_message.lower()
    
    # Smart query parsing - try to extract field and topic from one message
    if detect_complete_query(user_input):
        field, topic = extract_field_and_topic(user_input)
        if field and topic:
            return await search_and_respond(field, topic, session)
    
    # Ask for field if not detected
    session.awaiting_field = True
    return ChatResponse(
        response="I'll help you find research papers! What field are you interested in? (e.g., Computer Science, Physics, Biology, Mathematics)",
        action_type="info"
    )

def detect_complete_query(user_input: str) -> bool:
    """Detect if user provided both field and topic in one message"""
    field_keywords = {
        'computer science': ['machine learning', 'ai', 'neural networks', 'deep learning', 'nlp'],
        'physics': ['quantum', 'particle', 'relativity', 'thermodynamics'],
        'biology': ['genetics', 'evolution', 'molecular', 'neuroscience'],
        'mathematics': ['algebra', 'calculus', 'statistics', 'optimization']
    }
    
    for field, keywords in field_keywords.items():
        if any(keyword in user_input for keyword in keywords):
            return True
    return False

def extract_field_and_topic(user_input: str) -> tuple:
    """Extract field and topic from user input"""
    mappings = {
        'machine learning': ('Computer Science', 'machine learning'),
        'deep learning': ('Computer Science', 'deep learning'),
        'neural networks': ('Computer Science', 'neural networks'),
        'computer vision': ('Computer Science', 'computer vision'),
        'nlp': ('Computer Science', 'natural language processing'),
        'quantum': ('Physics', 'quantum mechanics'),
        'particle physics': ('Physics', 'particle physics'),
        'genetics': ('Biology', 'genetics'),
        'evolution': ('Biology', 'evolution'),
    }
    
    for keyword, (field, topic) in mappings.items():
        if keyword in user_input:
            return field, topic
    
    return None, None

async def handle_field_input(field: str, session: ConversationState) -> ChatResponse:
    """Handle field input from user"""
    session.current_field = field.strip()
    session.awaiting_field = False
    session.awaiting_topic = True
    
    suggestions = get_topic_suggestions(field.lower())
    
    return ChatResponse(
        response=f"Great! You've chosen {field}. Now specify a topic within this field.\n\nSome suggestions: {', '.join(suggestions)}",
        action_type="info"
    )

async def handle_topic_input(topic: str, session: ConversationState) -> ChatResponse:
    """Handle topic input and perform search"""
    session.current_topic = topic.strip()
    session.awaiting_topic = False
    
    return await search_and_respond(session.current_field, session.current_topic, session)

async def search_and_respond(field: str, topic: str, session: ConversationState) -> ChatResponse:
    """Search for papers and return response"""
    try:
        query = f"{field} {topic}"
        logger.info(f"Searching for: {query}")
        
        papers_info, has_more = search_papers(query, max_results=8)
        
        if not papers_info:
            session.awaiting_field = True  # Reset to ask for new field
            return ChatResponse(
                response=f"No papers found for '{topic}' in {field}. Try different keywords or a broader topic. What field would you like to search in?",
                action_type="search"
            )
        
        # Store papers for potential selection
        session.current_papers = papers_info
        session.awaiting_selection = True
        
        # Convert to Paper objects
        papers_list = []
        for pid, info in papers_info.items():
            papers_list.append(Paper(
                id=pid,
                title=info['title'],
                authors=info['authors'][:3],  # Limit authors for UI
                summary=info['summary'][:200] + "..." if len(info['summary']) > 200 else info['summary'],
                pdf_url=info['pdf_url'],
                published=info['published'],
                version=info['version'],
                status=info['status'],
                categories=info.get('categories', [])[:2]  # Limit categories
            ))
        
        # Create a detailed search summary like CLI
        total_available = len(papers_info)
        response_text = f"🔍 Search Results for '{topic}' in {field}\n\n"
        response_text += f"📄 Found {total_available} papers (showing top {len(papers_list)})\n\n"
        
        # Add paper list with details
        for i, paper in enumerate(papers_list, 1):
            # Extract year from published date
            year = paper.published.split('-')[0] if paper.published else 'Unknown'
            authors_text = ", ".join(paper.authors)
            if len(paper.authors) >= 3:
                authors_text += " et al."
            
            response_text += f"  {i}) [{year}] {paper.title}  \n"
            response_text += f"   👥 {authors_text}\n"
            response_text += f"   🏷️ {', '.join(paper.categories)} (ID: {paper.id})\n\n"
        
        if has_more:
            response_text += f"📊 Note: This is a subset of available papers. More results available.\n\n"
        
        response_text += "💡 Next Steps:\n"
        response_text += "• Tell me how many papers to download (e.g., '3 papers')\n"
        response_text += "• Or say 'new search' to try different keywords"
        
        return ChatResponse(
            response=response_text,
            papers=papers_list,
            action_type="search"
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return ChatResponse(
            response=f"Error searching for papers: {str(e)}. Please try again.",
            action_type="error"
        )

async def handle_selection_input(user_message: str, session: ConversationState, background_tasks: BackgroundTasks) -> ChatResponse:
    """Handle paper selection and download requests"""
    user_input = user_message.lower()
    
    if any(phrase in user_input for phrase in ["new search", "start over", "different topic"]):
        # Reset session
        session.__init__()
        return ChatResponse(
            response="Let's start fresh! What field are you interested in?",
            action_type="info"
        )
    
    if "download" in user_input or "get" in user_input:
        selected_papers = parse_paper_selection(user_message, session.current_papers)
        
        if not selected_papers:
            return ChatResponse(
                response="I couldn't understand your selection. Try:\n• 'Download papers 1,2,3' (by numbers)\n• 'Download papers by Smith' (by author)\n• 'Download all papers'",
                action_type="info"
            )
        
        # Start download in background
        background_tasks.add_task(
            download_papers_background,
            selected_papers,
            session.current_topic,
            session.current_field,
            session.current_papers
        )
        
        return ChatResponse(
            response=f"Starting download of {len(selected_papers)} papers! This may take a few minutes. The papers will be saved to your papers directory.\n\nWould you like to search for more papers?",
            action_type="download"
        )
    
    return ChatResponse(
        response="I'm ready to download papers for you! Just tell me which ones you'd like or say 'new search' to look for different papers.",
        action_type="info"
    )

def parse_paper_selection(user_message: str, papers_dict: dict) -> List[str]:
    """Parse user selection from message"""
    selected = []
    papers_list = list(papers_dict.items())
    user_input = user_message.lower()
    
    # Handle "all papers"
    if "all" in user_input:
        return list(papers_dict.keys())[:5]  # Limit to 5 for safety
    
    # Extract numbers
    numbers = re.findall(r'\b(\d+)\b', user_message)
    for num_str in numbers:
        idx = int(num_str)
        if 1 <= idx <= len(papers_list):
            selected.append(papers_list[idx-1][0])
    
    # Extract author names
    for pid, info in papers_list:
        for author in info['authors']:
            author_parts = author.lower().split()
            if any(part in user_input for part in author_parts if len(part) > 2):
                if pid not in selected:
                    selected.append(pid)
                    break
    
    return selected[:5]  # Limit to 5 papers

async def handle_paper_search(query: str, session: ConversationState, max_results: int = 10) -> ChatResponse:
    """Handle paper search for the new workflow with sequential numbering"""
    try:
        logger.info(f"New workflow search for: {query} (max_results: {max_results})")
        
        papers_info, has_more = search_papers(query, max_results=max_results)
        
        if not papers_info:
            return ChatResponse(
                response=f"No papers found for '{query}'. Please try different keywords.",
                action_type="search"
            )
        
        # Store papers for potential download
        session.current_papers.update(papers_info)
        
        # Add papers to global map with sequential numbering
        papers_list = []
        for pid, info in papers_info.items():
            # Assign sequential number
            serial_number = session.next_serial_number
            session.global_paper_map[serial_number] = pid
            session.next_serial_number += 1
            
            papers_list.append(Paper(
                id=pid,
                title=info['title'],
                authors=info['authors'][:3],
                summary=info['summary'][:200] + "..." if len(info['summary']) > 200 else info['summary'],
                pdf_url=info['pdf_url'],
                published=info['published'],
                version=info['version'],
                status=info['status'],
                categories=info.get('categories', [])[:2]
            ))
        
        # Determine the range of serial numbers
        start_num = session.next_serial_number - len(papers_list)
        end_num = session.next_serial_number - 1
        
        logger.info(f"Paper numbering: start_num={start_num}, end_num={end_num}, next_serial_number={session.next_serial_number}, papers_count={len(papers_list)}")
        
        response_text = f"Found {len(papers_list)} research papers! Here are papers {start_num}-{end_num}:\n\n"
        response_text += "Selected Papers:"
        
        return ChatResponse(
            response=response_text,
            papers=papers_list,
            action_type="search"
        )
        
    except Exception as e:
        logger.error(f"Paper search error: {e}")
        return ChatResponse(
            response=f"Error searching for papers: {str(e)}",
            action_type="error"
        )

async def handle_load_more_papers(query: str, session: ConversationState, max_results: int = 5) -> ChatResponse:
    """Handle loading more papers with sequential numbering by searching for additional papers"""
    try:
        logger.info(f"Searching for more papers on: {query} (max_results: {max_results})")
        
        # Try multiple search strategies to find new papers
        search_attempts = [
            query,  # Original query
            f"{query} recent",  # Add recent keyword
            f"{query} 2024",  # Add current year
            f"{query} 2023",  # Add previous year
            f"{query} latest",  # Add latest keyword
        ]
        
        new_papers = {}
        search_used = query
        
        # Try each search strategy until we find new papers
        for search_query in search_attempts:
            logger.info(f"Trying search: {search_query}")
            papers_info, has_more = search_papers(search_query, max_results=max_results)
            
            if papers_info:
                # Filter out papers that are already loaded to avoid duplicates
                for pid, info in papers_info.items():
                    if pid not in session.current_papers:
                        new_papers[pid] = info
                
                if new_papers:
                    search_used = search_query
                    logger.info(f"Found {len(new_papers)} new papers using search: {search_query}")
                    break
        
        # If still no new papers, try broader searches
        if not new_papers:
            query_parts = query.split()
            broader_searches = []
            if len(query_parts) > 1:
                broader_searches.append(f"{query_parts[0]} {query_parts[-1]}")  # First and last word
            if len(query_parts) > 0:
                broader_searches.append(query_parts[0])  # Just the first word (often field)
                if len(query_parts) > 1:
                    broader_searches.append(query_parts[-1])  # Just the last word (often topic)
            
            for search_query in broader_searches:
                logger.info(f"Trying broader search: {search_query}")
                papers_info, has_more = search_papers(search_query, max_results=max_results)
                
                if papers_info:
                    for pid, info in papers_info.items():
                        if pid not in session.current_papers:
                            new_papers[pid] = info
                    
                    if new_papers:
                        search_used = search_query
                        logger.info(f"Found {len(new_papers)} new papers using broader search: {search_query}")
                        break
        
        if not new_papers:
            return ChatResponse(
                response="No more relevant papers available on arXiv for this topic. All available papers have been loaded. You can now proceed to download selection or search for a different topic.",
                action_type="search"
            )
        
        # Store new papers for potential download
        session.current_papers.update(new_papers)
        
        # Add papers to global map with sequential numbering
        papers_list = []
        for pid, info in new_papers.items():
            # Assign sequential number
            serial_number = session.next_serial_number
            session.global_paper_map[serial_number] = pid
            session.next_serial_number += 1
            
            papers_list.append(Paper(
                id=pid,
                title=info['title'],
                authors=info['authors'][:3],
                summary=info['summary'][:200] + "..." if len(info['summary']) > 200 else info['summary'],
                pdf_url=info['pdf_url'],
                published=info['published'],
                version=info['version'],
                status=info['status'],
                categories=info.get('categories', [])[:2]
            ))
        
        # Determine the range of serial numbers
        start_num = session.next_serial_number - len(papers_list)
        end_num = session.next_serial_number - 1
        
        logger.info(f"Load more numbering: start_num={start_num}, end_num={end_num}, next_serial_number={session.next_serial_number}, papers_count={len(papers_list)}")
        
        response_text = f"Found {len(papers_list)} additional papers on the same topic! Here are papers {start_num}-{end_num}:\n\n"
        response_text += "Selected Papers:"
        
        return ChatResponse(
            response=response_text,
            papers=papers_list,
            action_type="search"
        )
        
    except Exception as e:
        logger.error(f"Load more papers error: {e}")
        return ChatResponse(
            response=f"Error searching for more papers: {str(e)}",
            action_type="error"
        )

async def generate_detailed_summaries(paper_ids: List[str], papers_info: dict) -> dict:
    """Generate concise 100-150 word summaries for selected papers"""
    detailed_summaries = {}
    
    for pid in paper_ids:
        if pid in papers_info:
            try:
                # Use the existing summarize_with_gemini function for detailed summaries
                from bot import summarize_with_gemini
                detailed_summary = await asyncio.get_event_loop().run_in_executor(
                    None, summarize_with_gemini, papers_info[pid]
                )
                detailed_summaries[pid] = detailed_summary
            except Exception as e:
                logger.error(f"Error generating detailed summary for {pid}: {e}")
                # Fallback to original summary if AI generation fails
                detailed_summaries[pid] = papers_info[pid]['summary']
    
    return detailed_summaries

async def handle_download_request(message: ChatMessage, session: ConversationState, background_tasks: BackgroundTasks) -> ChatResponse:
    """Handle download request with reference generation"""
    try:
        # Validate inputs
        if not message.paper_ids:
            logger.warning("Download request received with no paper IDs")
            return ChatResponse(
                response="No papers selected for download. Please select papers first.",
                action_type="error"
            )
        
        if not message.field or not message.topic:
            logger.warning(f"Download request missing field or topic: field={message.field}, topic={message.topic}")
            return ChatResponse(
                response="Missing required field or topic information for download.",
                action_type="error"
            )
        
        # Get papers info
        papers_info = session.current_papers or {}
        if not papers_info:
            logger.warning("No papers info available in session")
            return ChatResponse(
                response="No papers information available. Please search for papers first.",
                action_type="error"
            )
        
        selected_papers = {pid: papers_info[pid] for pid in message.paper_ids if pid in papers_info}
        
        if not selected_papers:
            logger.warning(f"No valid papers found for IDs: {message.paper_ids}")
            return ChatResponse(
                response="No valid papers found for download. Please ensure papers are loaded first.",
                action_type="error"
            )
        
        missing_papers = [pid for pid in message.paper_ids if pid not in papers_info]
        if missing_papers:
            logger.warning(f"Some papers missing from session: {missing_papers}")
        
        logger.info(f"Starting download: {len(selected_papers)} papers, field={message.field}, topic={message.topic}")
        
        # Generate detailed summaries for selected papers
        detailed_summaries = await generate_detailed_summaries(message.paper_ids, selected_papers)
        
        # Start download in background
        background_tasks.add_task(
            download_with_references,
            message.paper_ids,
            message.topic,
            message.field,
            selected_papers,
            detailed_summaries
        )
        
        # Return the detailed summaries to frontend for display during download
        return ChatResponse(
            response=f"📥 Starting download of {len(selected_papers)} papers with detailed summaries...",
            action_type="download",
            papers=[Paper(
                id=pid,
                title=selected_papers[pid]['title'],
                authors=selected_papers[pid]['authors'][:3],
                summary=detailed_summaries.get(pid, selected_papers[pid]['summary']),
                pdf_url=selected_papers[pid]['pdf_url'],
                published=selected_papers[pid]['published'],
                version=selected_papers[pid]['version'],
                status=selected_papers[pid]['status'],
                categories=selected_papers[pid].get('categories', [])[:2]
            ) for pid in message.paper_ids if pid in selected_papers]
        )
        
        
    except Exception as e:
        logger.error(f"Unexpected error in download request: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ChatResponse(
            response=f"Error processing download request: {str(e)}",
            action_type="error"
        )

async def download_with_references(paper_ids: List[str], topic: str, field: str, papers_info: dict, detailed_summaries: dict = None):
    """Background task for downloading papers and generating references"""
    try:
        logger.info(f"Starting download: {len(paper_ids)} papers")
        
        # Validate inputs
        if not paper_ids:
            logger.error("No paper IDs provided for download")
            return
        
        if not papers_info:
            logger.error("No papers info provided for download")
            return
        
        # Create downloads directory in project papers folder
        try:
            papers_dir = os.path.join(get_project_papers_directory(), safe_filename(field), safe_filename(topic))
            os.makedirs(papers_dir, exist_ok=True)
            logger.info(f"Created download directory: {papers_dir}")
        except Exception as e:
            logger.error(f"Failed to create download directory: {e}")
            return
        
        # Download papers (use web_mode=True for non-interactive downloads)
        try:
            # Use custom directory path by calling the function with a custom path
            successful_downloads = download_papers_to_path(paper_ids, topic, field, papers_info, papers_dir, web_mode=True)
            logger.info(f"Download completed: {len(successful_downloads)} successful downloads")
        except Exception as e:
            logger.error(f"Error during paper downloads: {e}")
            successful_downloads = []
        

        
        # Generate detailed summary file for web interface
        if successful_downloads:
            try:
                await generate_download_summary(paper_ids, papers_info, papers_dir, successful_downloads, detailed_summaries)
            except Exception as e:
                logger.error(f"Error generating download summary: {e}")
        
        logger.info("Download with references completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error in download_with_references: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")



async def generate_download_summary(paper_ids: List[str], papers_info: dict, papers_dir: str, successful_downloads: List[tuple], detailed_summaries: dict = None):
    """Generate a detailed download summary with AI summaries like CLI"""
    try:
        from bot import summarize_with_gemini
        
        summary_content = "📑 Downloaded Papers Summary\n"
        summary_content += "=" * 50 + "\n\n"
        
        for pid in paper_ids:
            if pid not in papers_info:
                continue
                
            paper = papers_info[pid]
            
            # Check if this paper was successfully downloaded
            downloaded = any(download_pid == pid for download_pid, _ in successful_downloads)
            if not downloaded:
                continue
            
            summary_content += f"📄 {paper['title']}\n"
            summary_content += f"👥 {', '.join(paper['authors'][:2])}{' et al.' if len(paper['authors']) > 2 else ''}\n"
            summary_content += f"📅 {paper['published']} | {paper['status']}\n"
            
            # Find the local file
            local_file = None
            for download_pid, file_path in successful_downloads:
                if download_pid == pid:
                    local_file = os.path.basename(file_path)
                    break
            
            if local_file:
                summary_content += f"💾 {local_file}\n"
            
            # Use detailed summary if available, otherwise generate AI summary
            if detailed_summaries and pid in detailed_summaries:
                summary_content += f"📝 Summary: {detailed_summaries[pid]}\n"
            else:
                try:
                    ai_summary = await asyncio.get_event_loop().run_in_executor(
                        None, summarize_with_gemini, paper
                    )
                    if ai_summary:
                        summary_content += f"📝 Summary: {ai_summary}\n"
                    else:
                        summary_content += f"📝 Abstract: {paper['summary'][:150]}...\n"
                except Exception as e:
                    logger.warning(f"Could not generate AI summary for {pid}: {e}")
                    summary_content += f"📝 Abstract: {paper['summary'][:150]}...\n"
            
            summary_content += "\n" + "-" * 50 + "\n\n"
        
        # Save summary to file
        summary_file = os.path.join(papers_dir, "download_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"Download summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error generating download summary: {e}")

def download_papers_to_path(paper_ids: List[str], topic: str, field: str, papers_info: dict, download_path: str, web_mode: bool = False) -> List[tuple]:
    """Download papers to a specific path"""
    try:
        from bot import download_single_paper_enhanced, load_metadata, save_metadata, MAX_CONCURRENT_DOWNLOADS
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime
        
        os.makedirs(download_path, exist_ok=True)
        logger.info(f"Downloading {len(paper_ids)} papers to {download_path}")
        
        successful_downloads = []
        failed_downloads = []
        skipped_downloads = []
        
        # Load and update metadata
        metadata = load_metadata()
        
        # Use ThreadPoolExecutor for controlled concurrency
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
            # Submit download tasks
            future_to_pid = {
                executor.submit(download_single_paper_enhanced, pid, papers_info[pid], download_path, True, web_mode): pid 
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
                    if not web_mode:
                        print(f"✅ Successfully downloaded: {pid}")
                elif status == "skipped":
                    skipped_downloads.append(pid)
                    if not web_mode:
                        print(f"⏭️  Skipped existing: {pid}")
                else:
                    failed_downloads.append((pid, status))
                    if not web_mode:
                        print(f"❌ Failed to download {pid}: {status}")
        
        # Save updated metadata
        save_metadata(metadata)
        
        if failed_downloads:
            logger.info(f"Failed downloads: {failed_downloads}")
        
        return successful_downloads
        
    except Exception as e:
        logger.error(f"Error in download_papers_to_path: {e}")
        return []



async def download_papers_background(paper_ids: List[str], topic: str, field: str, papers_info: dict):
    """Background task for downloading papers"""
    try:
        logger.info(f"Starting background download of {len(paper_ids)} papers")
        download_papers_enhanced(paper_ids, topic, field, papers_info)
        logger.info("Background download completed successfully")
    except Exception as e:
        logger.error(f"Background download error: {e}")

def get_topic_suggestions(field: str) -> List[str]:
    """Get topic suggestions based on field"""
    suggestions = {
        "computer science": ["machine learning", "artificial intelligence", "computer vision", "natural language processing", "algorithms", "deep learning", "neural networks", "distributed systems"],
        "physics": ["quantum mechanics", "particle physics", "astrophysics", "condensed matter", "quantum computing", "relativity", "cosmology", "optics"],
        "biology": ["genetics", "molecular biology", "neuroscience", "evolution", "bioinformatics", "cell biology", "immunology", "ecology"],
        "mathematics": ["algebra", "statistics", "optimization", "topology", "number theory", "differential equations", "graph theory", "probability"],
        "math": ["algebra", "statistics", "optimization", "topology", "number theory", "differential equations", "graph theory", "probability"],
        "chemistry": ["organic chemistry", "materials science", "biochemistry", "catalysis", "quantum chemistry", "analytical chemistry", "physical chemistry", "polymer science"],
        "medicine": ["oncology", "cardiology", "neurology", "immunology", "pharmacology", "epidemiology", "surgery", "radiology"],
        "engineering": ["mechanical engineering", "electrical engineering", "civil engineering", "chemical engineering", "biomedical engineering", "robotics", "control systems", "signal processing"]
    }
    
    field_lower = field.lower()
    for key, topics in suggestions.items():
        if key in field_lower or any(word in field_lower for word in key.split()):
            return topics
    
    return ["machine learning", "quantum computing", "gene therapy", "optimization", "neural networks"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level="info"
    )