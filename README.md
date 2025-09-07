# Research Assistant

A modern web application that helps researchers find, analyze, and download academic papers from arXiv using AI-powered summaries.

## 🚀 Features

- **Smart Paper Search**: Find relevant research papers by field and topic
- **Sequential Numbering**: Papers are numbered sequentially (1, 2, 3...) for easy reference
- **Load More Papers**: Continuously search for additional papers on the same topic
- **AI-Powered Summaries**: Get concise 100-150 word summaries of papers
- **Batch Download**: Download multiple papers with detailed summaries
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS
- **Real-time Progress**: Live download progress with paper details

## 🏗️ Architecture

- **Frontend**: Next.js 15 with TypeScript and Tailwind CSS
- **Backend**: FastAPI with Python 3.11
- **AI Integration**: Google Gemini API for paper summarization
- **Paper Source**: ArXiv API for academic papers
- **Deployment**: Railway (backend) + Vercel (frontend)

## 📁 Project Structure

```
researchbot/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── bot.py              # Core logic (search, AI, download)
│   ├── requirements.txt    # Python dependencies
│   ├── Procfile           # Railway deployment
│   ├── Dockerfile         # Docker configuration
│   └── railway.json       # Railway settings
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app directory
│   │   ├── api/           # API routes
│   │   ├── page.tsx       # Main chat interface
│   │   └── globals.css    # Global styles
│   ├── package.json       # Node.js dependencies
│   └── vercel.json        # Vercel deployment
├── DEPLOYMENT.md          # Deployment guide
└── README.md              # This file
```

## 🛠️ Local Development

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Gemini API key

### Backend Setup

1. Navigate to backend directory:

   ```bash
   cd backend
   ```

2. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   cp env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

5. Run the backend:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Set up environment variables:

   ```bash
   cp env.local.example .env.local
   # Edit .env.local and set BACKEND_URL
   ```

4. Run the frontend:

   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000)

## 🚀 Deployment

### Quick Deploy

1. **Backend (Railway)**:

   - Connect GitHub repo
   - Select `backend` folder
   - Add environment variables
   - Deploy

2. **Frontend (Vercel)**:
   - Connect GitHub repo
   - Select `frontend` folder
   - Add environment variables
   - Deploy

### Detailed Instructions

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment guide.

## 🔧 Environment Variables

### Backend (Railway)

```
GEMINI_API_KEY=your_gemini_api_key
ENVIRONMENT=production
FRONTEND_URL=https://your-vercel-app.vercel.app
PORT=8000
LOG_LEVEL=INFO
```

### Frontend (Vercel)

```
BACKEND_URL=https://your-railway-app.railway.app
NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app
```

## 📖 Usage

1. **Start a Search**: Enter your research field and topic
2. **Browse Papers**: View 10 papers with sequential numbering
3. **Load More**: Get additional papers on the same topic
4. **Select Papers**: Choose papers by number, author, or year
5. **Download**: Get papers with AI-generated summaries

## 🎯 Key Features Explained

### Sequential Paper Numbering

- Papers are numbered 1, 2, 3, 4, 5... regardless of when they're loaded
- Numbers remain fixed once assigned
- Load more papers continues numbering (6, 7, 8, 9, 10...)

### AI Summaries

- 100-150 word concise summaries
- Generated using Google Gemini API
- Include key findings, methodology, and significance

### Smart Search

- Multiple search strategies for finding more papers
- Duplicate prevention
- Continuous loading until no more papers available

## 🛡️ Security

- No API keys in code
- Environment variables for all secrets
- CORS properly configured
- Input validation and sanitization

## 📊 Performance

- Concurrent paper downloads
- Background processing
- Optimized API calls
- Responsive UI with real-time feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter issues:

1. Check the [deployment guide](./DEPLOYMENT.md)
2. Verify environment variables
3. Check logs in Railway/Vercel
4. Ensure Gemini API key is valid

## 🔄 Updates

- **v1.0.0**: Initial release with core functionality
- Sequential paper numbering
- AI-powered summaries
- Batch download system
- Modern UI/UX

---

**Happy Researching! 🔬📚**
