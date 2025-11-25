# WADL Agentic Recruiter

Automated candidate matching system.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
LLM_API_KEY=your_openai_key
SEARCH_API_KEY=your_serpapi_key
BRIGHTDATA_API_KEY=your_brightdata_key
BRIGHTDATA_DATASET_ID=gd_l1viktl72bvl7bjuj0
```

```bash
python -m scripts.init_db
python -m scripts.import_placements
streamlit run src/ui/app.py
```


