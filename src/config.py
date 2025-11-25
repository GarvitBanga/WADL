from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
import os

def get_streamlit_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    search_api_key: Optional[str] = None
    bing_api_key: Optional[str] = None
    database_url: str = "sqlite:///./data/wadl_recruiter.db"
    max_concurrent_requests: int = 10
    proxy_url: Optional[str] = None
    proxy_file: Optional[str] = None
    brightdata_username: Optional[str] = None
    brightdata_password: Optional[str] = None
    brightdata_endpoint: Optional[str] = None
    brightdata_api_key: Optional[str] = None
    brightdata_dataset_id: Optional[str] = None
    scraperapi_key: Optional[str] = None
    skip_html_fetch: bool = False
    use_staffspy: bool = False
    
    def get_proxy_list(self) -> List[str]:
        proxies = []
        if self.proxy_url:
            proxies.extend([p.strip() for p in self.proxy_url.split(",") if p.strip()])
        if self.proxy_file:
            proxy_file_path = Path(self.proxy_file)
            if not proxy_file_path.is_absolute():
                project_root = Path(__file__).parent.parent
                possible_paths = [
                    project_root / proxy_file_path,
                    project_root.parent / proxy_file_path,
                    Path.cwd() / proxy_file_path,
                ]
                proxy_file_path = None
                for path in possible_paths:
                    if path.exists():
                        proxy_file_path = path
                        break
            if proxy_file_path and proxy_file_path.exists():
                try:
                    with open(proxy_file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if ':' in line:
                                    if not line.startswith(('http://', 'https://', 'socks5://')):
                                        line = f"http://{line}"
                                    proxies.append(line)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load proxy file: {e}")
        return proxies
    
    model_config = SettingsConfigDict(
        env_file=str(env_path) if env_path.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()

settings.llm_api_key = settings.llm_api_key or get_streamlit_secret("LLM_API_KEY", "")
settings.search_api_key = settings.search_api_key or get_streamlit_secret("SEARCH_API_KEY") or None
settings.brightdata_api_key = settings.brightdata_api_key or get_streamlit_secret("BRIGHTDATA_API_KEY") or None
settings.brightdata_dataset_id = settings.brightdata_dataset_id or get_streamlit_secret("BRIGHTDATA_DATASET_ID") or None
settings.scraperapi_key = settings.scraperapi_key or get_streamlit_secret("SCRAPERAPI_KEY") or None

model = get_streamlit_secret("LLM_MODEL", "")
if model:
    settings.llm_model = model

model = get_streamlit_secret("EMBEDDING_MODEL", "")
if model:
    settings.embedding_model = model

db_url = get_streamlit_secret("DATABASE_URL", "")
if db_url:
    settings.database_url = db_url
elif settings.database_url == "sqlite:///./data/wadl_recruiter.db":
    import tempfile
    db_path = Path(tempfile.gettempdir()) / "wadl_recruiter.db"
    settings.database_url = f"sqlite:///{db_path}"

max_conc = get_streamlit_secret("MAX_CONCURRENT_REQUESTS", "")
if max_conc:
    try:
        settings.max_concurrent_requests = int(max_conc)
    except:
        pass

