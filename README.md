# Research Assistant Chatbot

An intelligent research assistant that helps you find and download academic papers from arXiv with an improved guided workflow.

## Features

### 🚀 Improved Workflow

- **Guided Experience**: Step-by-step process from greeting to completion
- **Field Selection**: Choose your research field with suggestions
- **Topic Selection**: Specify your research topic with field-specific suggestions
- **Paper Listing**: Browse found papers with detailed information
- **Download Count**: Select how many papers to download
- **Reference Format**: Choose between APA, MLA, IEEE, or no references
- **Real-time Feedback**: See download progress with paper summaries
- **Downloads Directory**: Files saved directly to your Downloads/papers folder

### 🎨 Modern UI

- **Dark/Light Theme**: Toggle between themes
- **Progress Tracker**: Visual workflow progress indicator
- **Responsive Design**: Works on desktop and mobile
- **Paper Cards**: Enhanced paper display with categories and summaries
- **Restart Functionality**: Start new searches with one click

### 📚 Advanced Features

- **Reference Generation**: Automatic citation generation in multiple formats
- **Smart Search**: Powered by arXiv and Gemini AI
- **Paper Summaries**: AI-generated summaries during download
- **Download Management**: Files organized by field and topic
- **Error Handling**: Robust error handling and user feedback

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Gemini AI API key

### Backend Setup

1. Navigate to the backend directory:

```bash
cd backend
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory:

```env
GENAI_API_KEY=your_gemini_api_key_here
```

5. Start the backend server:

```bash
python main.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The frontend will run on `http://localhost:3000`

## Usage Guide

### Step-by-Step Workflow

1. **Welcome**: The assistant greets you and introduces its capabilities
2. **Field Selection**: Enter your research field (e.g., "Computer Science", "Physics")
3. **Topic Selection**: Specify your topic with helpful suggestions
4. **Paper Discovery**: Browse the found papers with detailed information
5. **Download Count**: Choose how many papers to download (1-8)
6. **Reference Format**: Select APA, MLA, IEEE, or none
7. **Download Process**: Watch real-time progress with paper summaries
8. **Completion**: Files saved to Downloads/papers folder with optional references

### Features in Detail

#### Progress Tracker

- Visual indicator showing current step
- Completed steps marked in green
- Current step highlighted in blue

#### Paper Cards

- Numbered papers for easy reference
- Author information with "et al." for multiple authors
- Publication year and status
- Research categories/tags
- Brief summary preview
- Direct PDF link

#### Download Organization

```
Downloads/
└── papers/
    └── Computer_Science/
        └── machine_learning/
            ├── paper1.pdf
            ├── paper2.pdf
            └── references_apa.txt
```

#### Reference Formats

**APA Style**:

```
Smith, J. A., & Doe, J. B. (2023). Deep Learning in Computer Vision. arXiv preprint arXiv:2301.12345.
```

**MLA Style**:

```
Smith, John A. et al. "Deep Learning in Computer Vision." arXiv, 2023, arXiv:2301.12345.
```

**IEEE Style**:

```
J. A. Smith and J. B. Doe, "Deep Learning in Computer Vision," arXiv preprint arXiv:2301.12345, 2023.
```

### Restart and Continue

- Click the search icon in the header to start a new search
- Type "yes" after completion to begin another search
- Theme preferences persist across sessions

## API Endpoints

### POST /api/chat

Main chat endpoint supporting the workflow.

**Request Body**:

```json
{
  "message": "machine learning",
  "workflow_step": "search",
  "session_id": "optional",
  "paper_ids": ["2301.12345"],
  "reference_format": "APA",
  "field": "Computer Science",
  "topic": "machine learning"
}
```

**Response**:

```json
{
  "response": "Found 8 papers!",
  "papers": [...],
  "action_type": "search",
  "session_id": "default"
}
```

## File Structure

```
researchbot/
├── backend/
│   ├── main.py           # FastAPI server with workflow logic
│   ├── bot.py            # Core research functions
│   ├── requirements.txt  # Python dependencies
│   └── papers/          # Downloaded papers (local fallback)
├── frontend/
│   ├── app/
│   │   ├── page.tsx     # Main React component with workflow UI
│   │   ├── globals.css  # Global styles and animations
│   │   └── api/chat/route.ts  # API proxy to backend
│   ├── package.json     # Node.js dependencies
│   └── tailwind.config.ts  # Tailwind configuration
└── README.md           # This file
```

## Troubleshooting

### Backend Issues

- Ensure `GENAI_API_KEY` is set in the `.env` file
- Check that port 8000 is available
- Verify Python dependencies are installed in the virtual environment

### Frontend Issues

- Ensure Node.js version is 16 or higher
- Check that port 3000 is available
- Verify the backend is running on port 8000

### Download Issues

- Check Downloads directory permissions
- Ensure internet connection for arXiv access
- Verify sufficient disk space

### API Rate Limits

- Gemini API has rate limits for summary generation
- The system includes automatic retry logic with exponential backoff
- Large download batches may take time due to rate limiting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review the console logs for error details
3. Ensure all dependencies are properly installed
4. Verify API keys and environment setup
