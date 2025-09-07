# Deployment Guide

This guide will help you deploy the Research Assistant project to Railway (backend) and Vercel (frontend).

## Prerequisites

- GitHub account
- Railway account (https://railway.app)
- Vercel account (https://vercel.com)
- Gemini API key

## Backend Deployment (Railway)

### 1. Prepare Backend Repository

1. Push your code to GitHub
2. Ensure all files are committed:
   - `backend/main.py`
   - `backend/bot.py`
   - `backend/requirements.txt`
   - `backend/Procfile`
   - `backend/runtime.txt`
   - `backend/railway.json`
   - `backend/env.example`

### 2. Deploy to Railway

1. Go to [Railway](https://railway.app) and sign in
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Choose the `backend` folder as the root directory
5. Railway will automatically detect it's a Python project

### 3. Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
ENVIRONMENT=production
FRONTEND_URL=https://your-vercel-app.vercel.app
PORT=8000
LOG_LEVEL=INFO
```

### 4. Get Backend URL

After deployment, Railway will provide a URL like:
`https://your-app-name.railway.app`

## Frontend Deployment (Vercel)

### 1. Prepare Frontend Repository

Ensure your frontend code is in the `frontend` folder with:

- `package.json`
- `vercel.json`
- `env.local.example`

### 2. Deploy to Vercel

1. Go to [Vercel](https://vercel.com) and sign in
2. Click "New Project" → "Import Git Repository"
3. Select your repository
4. Set the root directory to `frontend`
5. Vercel will automatically detect it's a Next.js project

### 3. Configure Environment Variables

In Vercel dashboard, go to your project → Settings → Environment Variables and add:

```
BACKEND_URL=https://your-app-name.railway.app
NEXT_PUBLIC_API_URL=https://your-app-name.railway.app
```

### 4. Deploy

Click "Deploy" and wait for the build to complete.

## Post-Deployment

### 1. Update CORS Settings

In Railway, update the `FRONTEND_URL` environment variable to your actual Vercel URL.

### 2. Test the Application

1. Visit your Vercel URL
2. Try searching for papers
3. Test the download functionality

## Environment Variables Summary

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

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure `FRONTEND_URL` in Railway matches your Vercel URL exactly
2. **API Connection Failed**: Check that `BACKEND_URL` in Vercel is correct
3. **Gemini API Errors**: Verify your API key is correct and has sufficient quota
4. **Build Failures**: Check the build logs in Railway/Vercel for specific errors

### Logs

- **Railway**: Go to your project → Deployments → View logs
- **Vercel**: Go to your project → Functions → View function logs

## File Structure

```
researchbot/
├── backend/
│   ├── main.py
│   ├── bot.py
│   ├── requirements.txt
│   ├── Procfile
│   ├── runtime.txt
│   ├── railway.json
│   └── env.example
├── frontend/
│   ├── app/
│   ├── package.json
│   ├── vercel.json
│   └── env.local.example
└── DEPLOYMENT.md
```

## Support

If you encounter issues:

1. Check the logs in both Railway and Vercel
2. Verify all environment variables are set correctly
3. Ensure your Gemini API key is valid and has quota
4. Check that both services are running and accessible
