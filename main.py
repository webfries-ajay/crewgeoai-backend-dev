from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from core.config import settings
from core.database import init_db, close_db
from routers import auth, admin_auth, projects, files

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting up CrewGeoAI Backend...")
    await init_db()
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CrewGeoAI Backend...")
    await close_db()
    logger.info("Database connections closed")

app = FastAPI(
    title="CrewGeoAI Backend",
    description="FastAPI backend for conversational AI and GeoAI platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin_auth.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(files.router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"message": "Hello World from CrewGeoAI Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "crewgeoai-backend"}

@app.get("/api/v1/hello")
async def hello_api():
    return {
        "message": "Hello from CrewGeoAI API!",
        "version": "1.0.0",
        "status": "active"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 