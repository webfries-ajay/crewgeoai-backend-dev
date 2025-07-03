from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os

class Settings(BaseSettings):
    # App Configuration
    app_name: str = "CrewGeoAI Backend"
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    
    # Database Configuration
    database_url: str = "postgresql+asyncpg://localhost:5432/crewgeoai"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 10
    
    # JWT Configuration
    secret_key: str = "your-super-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Password Hashing
    bcrypt_rounds: int = 12
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001", 
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003"
    ]
    
    # Rate Limiting
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # seconds
    
    # AI API Configuration
    openai_api_key: str = ""
    google_api_key: str = ""
    
    # File Storage
    max_file_size: int = 1073741824  # 1GB
    upload_dir: str = "./uploads"
    
    # Additional AI/Processing Configuration (allow extra fields)
    openai_model: str = "gpt-4o"
    max_image_size: str = "8192"
    jpeg_quality: str = "98"
    max_file_size_mb: str = "30"
    
    # Analysis Configuration
    defect_detection_mode: str = "true"
    max_defect_tokens: str = "3000"
    enable_tiff_chunking: str = "true"
    chunk_size: str = "1024"
    chunk_overlap: str = "512"
    max_chunks_to_analyze: str = "25"
    
    # Metadata Extraction
    extract_orthomosaic_metadata: str = "true"
    extract_gps_coordinates: str = "true"
    extract_flight_parameters: str = "true"
    extract_camera_settings: str = "true"
    extract_coordinate_system: str = "true"
    show_geospatial_info: str = "true"
    
    # Spatial Analysis
    scale_analysis: str = "true"
    resolution_analysis: str = "true"
    coordinate_system_detection: str = "true"
    spatial_reference_extraction: str = "true"
    projection_info: str = "true"
    ground_sample_distance: str = "true"
    
    # Processing Options
    combine_chunk_results: str = "true"
    analyze_all_chunks: str = "false"
    detailed_thermal_reporting: str = "true"
    orthomosaic_reporting: str = "true"
    metadata_detailed_output: str = "true"
    
    # Pydantic v2 configuration
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from environment
    )

# Create global settings instance
settings = Settings() 