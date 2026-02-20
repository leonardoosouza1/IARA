import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from dotenv import load_dotenv

from src.database.models import Base

load_dotenv()

logger = logging.getLogger("Database")

# DB Configuration
DATABASE_URL = os.getenv("DB_URI", "sqlite+aiosqlite:///./saci_core.db")

# Patch driver for AsyncPG if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# If running outside Docker, 'db' host might not resolve.
# We trust the user OR we could have a fallback. 
# For now, we trust the env or the default SQLite fallback if env is missing.

engine = create_async_engine(
    DATABASE_URL,
    echo=False, # Set to True for SQL debug
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def init_db():
    """Create tables if they don't exist."""
    try:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.drop_all) # Dangerous!
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database Tables Initialized (Trades, Signals)")
    except Exception as e:
        logger.error(f"❌ Database Init Failed: {e}")
        # Fallback for dev without DB?
        # Only if we want to allow running without DB (Mock mode)

async def get_db():
    """Dependency for FastAPI/Context Managers."""
    async with AsyncSessionLocal() as session:
        yield session
