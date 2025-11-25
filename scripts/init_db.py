from src.db.session import engine
from src.db.models import Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)

