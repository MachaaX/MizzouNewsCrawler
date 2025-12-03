import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import DatabaseManager
from sqlalchemy import text

db = DatabaseManager()
with db.get_session() as session:
    count = session.execute(text("SELECT COUNT(*) FROM articles WHERE status = 'labeled'")).scalar()
    print(f"Total labeled articles: {count:,}")
