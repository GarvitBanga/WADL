from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from src.config import settings
from src.db.models import Base

db_url = settings.database_url
if db_url.startswith("sqlite:///"):
    db_path_str = db_url.replace("sqlite:///", "")
    if db_path_str != ":memory:":
        db_path = Path(db_path_str)
        db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False
)

def init_db():
    import src.db.models
    Base.metadata.create_all(bind=engine, checkfirst=True)

init_db()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

