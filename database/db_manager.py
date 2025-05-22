from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# Verbindung zur Datenbank herstellen
DATABASE_URL = "sqlite:///database.db"  # Pfad zur SQLite-Datenbank

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Tabellen erstellen
def create_tables():
    Base.metadata.create_all(bind=engine)
