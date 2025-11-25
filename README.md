# WADL Agentic Recruiter

Automated candidate matching system that sources, enriches, and ranks candidates based on job descriptions using historical placement data for calibration.

## Features

- Automatic JD parsing and requirement extraction
- Agentic sourcing with query generation and refinement
- Profile enrichment from LinkedIn using Bright Data API
- Placement-calibrated ranking engine
- Human-readable explanations for each candidate

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
LLM_API_KEY=your_openai_key
SEARCH_API_KEY=your_serpapi_key
BRIGHTDATA_API_KEY=your_brightdata_key
BRIGHTDATA_DATASET_ID=gd_l1viktl72bvl7bjuj0
```

3. Initialize database:
```bash
python -m scripts.init_db
python -m scripts.import_placements
```

4. Run the app:
```bash
streamlit run src/ui/app.py
```


