import pandas as pd
from pathlib import Path
from src.db.session import SessionLocal
from src.db.models import Placement

EXCEL_PATH = Path(__file__).parent.parent.parent / "Placements 2000-2025.xlsx"

def import_placements():
    df = pd.read_excel(EXCEL_PATH)
    session = SessionLocal()
    try:
        records = []
        for _, row in df.iterrows():
            p = Placement(
                name=str(row.get("Name", "")).strip(),
                company=str(row.get("Company", "")).strip(),
                job_title=str(row.get("Job Title", "")).strip(),
                position_id=str(row.get("Position Id", "")).strip() if "Position Id" in df.columns and pd.notna(row.get("Position Id")) else None,
                placement_type=str(row.get("Placement Type")) if "Placement Type" in df.columns and pd.notna(row.get("Placement Type")) else None,
                date_posted=str(row.get("Date Posted")) if "Date Posted" in df.columns and pd.notna(row.get("Date Posted")) else None,
                placement_date=str(row.get("Placement Date")) if "Placement Date" in df.columns and pd.notna(row.get("Placement Date")) else None,
                start_date=str(row.get("Start Date")) if "Start Date" in df.columns and pd.notna(row.get("Start Date")) else None,
            )
            records.append(p)
        session.bulk_save_objects(records)
        session.commit()
    finally:
        session.close()

if __name__ == "__main__":
    import_placements()

