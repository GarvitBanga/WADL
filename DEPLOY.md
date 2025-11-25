# Deployment Guide

## Quick Deploy to Streamlit Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `src/ui/app.py`
   - Click "Deploy"

3. **Add Secrets (Environment Variables):**
   In Streamlit Cloud → Settings → Secrets, add:
   ```
   LLM_API_KEY=your_openai_key
   SEARCH_API_KEY=your_serpapi_key
   BRIGHTDATA_API_KEY=your_brightdata_key
   BRIGHTDATA_DATASET_ID=gd_l1viktl72bvl7bjuj0
   ```

## Alternative: Deploy to Railway/Render/Fly.io

All support Python apps. Use `src/ui/app.py` as entry point.

## Local Testing Before Deploy

```bash
streamlit run src/ui/app.py
```

Make sure your `.env` file has all required keys.

