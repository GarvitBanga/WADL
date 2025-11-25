from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

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

