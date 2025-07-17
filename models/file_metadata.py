from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, BigInteger, ForeignKey, Float, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from core.database import Base

class FileMetadata(Base):
    """Comprehensive file metadata model with foreign key to File"""
    __tablename__ = "file_metadata"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    
    # Core metadata fields
    camera_make = Column(String(100), nullable=True)
    camera_model = Column(String(100), nullable=True)
    lens_model = Column(String(100), nullable=True)
    
    # Technical settings
    focal_length = Column(Float, nullable=True)
    aperture = Column(Float, nullable=True)  # f-stop value
    shutter_speed = Column(String(20), nullable=True)  # e.g., "1/250"
    iso_speed = Column(Integer, nullable=True)
    
    # GPS data
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)  # meters
    gps_precision = Column(Float, nullable=True)  # DOP value
    has_gps = Column(Boolean, default=False, index=True)
    
    # Date/time information
    date_taken = Column(DateTime(timezone=True), nullable=True)
    date_modified = Column(DateTime(timezone=True), nullable=True)
    timezone_offset = Column(String(10), nullable=True)  # e.g., "+05:30"
    
    # Image technical details
    color_space = Column(String(20), nullable=True)  # sRGB, Adobe RGB, etc.
    white_balance = Column(String(50), nullable=True)
    flash_used = Column(Boolean, nullable=True)
    exposure_mode = Column(String(50), nullable=True)
    metering_mode = Column(String(50), nullable=True)
    
    # Quality and processing
    image_quality = Column(String(20), nullable=True)  # RAW, JPEG, etc.
    compression_ratio = Column(Float, nullable=True)
    bit_depth = Column(Integer, nullable=True)
    
    # Professional equipment detection
    is_professional_grade = Column(Boolean, default=False, index=True)
    equipment_category = Column(String(50), nullable=True)  # drone, dslr, mirrorless, etc.
    
    # Content analysis flags
    has_faces = Column(Boolean, nullable=True)
    has_text = Column(Boolean, nullable=True)
    has_vehicles = Column(Boolean, nullable=True)
    has_buildings = Column(Boolean, nullable=True)
    has_vegetation = Column(Boolean, nullable=True)
    
    # Metadata completeness and quality
    metadata_completeness_score = Column(Float, default=0.0)  # 0-100 score
    extraction_confidence = Column(Float, default=0.0)  # 0-100 confidence
    
    # Raw metadata storage
    raw_exif_data = Column(JSONB, default=dict)
    raw_iptc_data = Column(JSONB, default=dict)
    raw_xmp_data = Column(JSONB, default=dict)
    processed_metadata = Column(JSONB, default=dict)
    
    # LLM integration
    llm_ready = Column(Boolean, default=False, index=True)
    token_count_estimate = Column(Integer, nullable=True)
    metadata_summary = Column(Text, nullable=True)  # Human-readable summary
    
    # Processing information
    extraction_method = Column(String(50), nullable=True)  # exiftool, pillow, etc.
    processing_time_ms = Column(Integer, nullable=True)
    extraction_errors = Column(JSONB, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    file = relationship("File", back_populates="file_metadata_record")
    
    def __repr__(self):
        return f"<FileMetadata(id={self.id}, file_id={self.file_id}, completeness={self.metadata_completeness_score})>"
    
    @classmethod
    def create_from_extraction(cls, file_id: str, metadata_dict: Dict[str, Any]) -> 'FileMetadata':
        """Create FileMetadata instance from extracted metadata dictionary"""
        metadata = cls(file_id=file_id)
        metadata.update_from_extraction(metadata_dict)
        return metadata
    
    def update_from_extraction(self, metadata_dict: Dict[str, Any]) -> None:
        """Update metadata fields from extraction dictionary"""
        # Camera information
        self.camera_make = metadata_dict.get('camera_make')
        self.camera_model = metadata_dict.get('camera_model')
        self.lens_model = metadata_dict.get('lens_model')
        
        # Technical settings
        self.focal_length = metadata_dict.get('focal_length')
        self.aperture = metadata_dict.get('aperture')
        self.shutter_speed = metadata_dict.get('shutter_speed')
        self.iso_speed = metadata_dict.get('iso_speed')
        
        # GPS data
        gps_data = metadata_dict.get('gps', {})
        if gps_data:
            self.latitude = gps_data.get('latitude')
            self.longitude = gps_data.get('longitude')
            self.altitude = gps_data.get('altitude')
            self.gps_precision = gps_data.get('precision')
            self.has_gps = bool(self.latitude and self.longitude)
        
        # Date/time
        self.date_taken = metadata_dict.get('date_taken')
        self.date_modified = metadata_dict.get('date_modified')
        self.timezone_offset = metadata_dict.get('timezone_offset')
        
        # Image details
        self.color_space = metadata_dict.get('color_space')
        self.white_balance = metadata_dict.get('white_balance')
        self.flash_used = metadata_dict.get('flash_used')
        self.exposure_mode = metadata_dict.get('exposure_mode')
        self.metering_mode = metadata_dict.get('metering_mode')
        
        # Quality
        self.image_quality = metadata_dict.get('image_quality')
        self.compression_ratio = metadata_dict.get('compression_ratio')
        self.bit_depth = metadata_dict.get('bit_depth')
        
        # Professional detection
        self.is_professional_grade = metadata_dict.get('is_professional_grade', False)
        self.equipment_category = metadata_dict.get('equipment_category')
        
        # Content flags
        self.has_faces = metadata_dict.get('has_faces')
        self.has_text = metadata_dict.get('has_text')
        self.has_vehicles = metadata_dict.get('has_vehicles')
        self.has_buildings = metadata_dict.get('has_buildings')
        self.has_vegetation = metadata_dict.get('has_vegetation')
        
        # Raw data
        self.raw_exif_data = metadata_dict.get('raw_exif_data', {})
        self.raw_iptc_data = metadata_dict.get('raw_iptc_data', {})
        self.raw_xmp_data = metadata_dict.get('raw_xmp_data', {})
        self.processed_metadata = metadata_dict.get('processed_metadata', {})
        
        # Quality scores
        self.metadata_completeness_score = metadata_dict.get('completeness_score', 0.0)
        self.extraction_confidence = metadata_dict.get('extraction_confidence', 0.0)
        
        # Processing info
        self.extraction_method = metadata_dict.get('extraction_method')
        self.extraction_errors = metadata_dict.get('extraction_errors', [])
        
        # Generate summary and token estimate
        self._generate_metadata_summary()
        self._calculate_token_estimate()
        self.llm_ready = True
    
    def _generate_metadata_summary(self) -> None:
        """Generate human-readable metadata summary"""
        summary_parts = []
        
        # Camera info
        if self.camera_make and self.camera_model:
            summary_parts.append(f"Camera: {self.camera_make} {self.camera_model}")
        
        # Technical settings
        tech_parts = []
        if self.focal_length:
            tech_parts.append(f"{self.focal_length}mm")
        if self.aperture:
            tech_parts.append(f"f/{self.aperture}")
        if self.shutter_speed:
            tech_parts.append(f"{self.shutter_speed}s")
        if self.iso_speed:
            tech_parts.append(f"ISO {self.iso_speed}")
        
        if tech_parts:
            summary_parts.append("Settings: " + ", ".join(tech_parts))
        
        # Location
        if self.has_gps:
            summary_parts.append(f"Location: {self.latitude:.6f}, {self.longitude:.6f}")
            if self.altitude:
                summary_parts.append(f"Altitude: {self.altitude}m")
        
        # Date
        if self.date_taken:
            summary_parts.append(f"Captured: {self.date_taken.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Equipment grade
        if self.is_professional_grade:
            summary_parts.append("Professional equipment")
        
        self.metadata_summary = " | ".join(summary_parts) if summary_parts else "Limited metadata available"
    
    def _calculate_token_estimate(self) -> None:
        """Calculate estimated token count for LLM consumption"""
        # Base estimate for structured metadata
        base_tokens = 50
        
        # Add tokens for each field with data
        field_tokens = 0
        for field_name in ['camera_make', 'camera_model', 'lens_model', 'color_space', 
                          'white_balance', 'exposure_mode', 'metering_mode', 'image_quality']:
            if getattr(self, field_name):
                field_tokens += 5
        
        # GPS data adds more context
        if self.has_gps:
            field_tokens += 20
        
        # Raw data contribution
        raw_data_size = len(str(self.raw_exif_data)) + len(str(self.raw_iptc_data)) + len(str(self.raw_xmp_data))
        raw_tokens = min(raw_data_size // 4, 200)  # Rough estimate, capped at 200
        
        self.token_count_estimate = base_tokens + field_tokens + raw_tokens
    
    def get_complete_metadata_json(self) -> Dict[str, Any]:
        """Get complete metadata formatted for LLM consumption"""
        return {
            "file_id": str(self.file_id),
            "summary": self.metadata_summary,
            "camera_info": {
                "make": self.camera_make,
                "model": self.camera_model,
                "lens": self.lens_model,
                "is_professional": self.is_professional_grade,
                "category": self.equipment_category
            },
            "technical_settings": {
                "focal_length": self.focal_length,
                "aperture": self.aperture,
                "shutter_speed": self.shutter_speed,
                "iso_speed": self.iso_speed,
                "color_space": self.color_space,
                "white_balance": self.white_balance,
                "flash_used": self.flash_used,
                "exposure_mode": self.exposure_mode,
                "metering_mode": self.metering_mode
            },
            "location_data": {
                "has_gps": self.has_gps,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
                "precision": self.gps_precision
            } if self.has_gps else None,
            "capture_time": {
                "date_taken": self.date_taken.isoformat() if self.date_taken else None,
                "timezone_offset": self.timezone_offset
            },
            "content_analysis": {
                "has_faces": self.has_faces,
                "has_text": self.has_text,
                "has_vehicles": self.has_vehicles,
                "has_buildings": self.has_buildings,
                "has_vegetation": self.has_vegetation
            },
            "quality_metrics": {
                "completeness_score": self.metadata_completeness_score,
                "extraction_confidence": self.extraction_confidence,
                "token_count_estimate": self.token_count_estimate
            },
            "raw_metadata": {
                "exif": self.raw_exif_data,
                "iptc": self.raw_iptc_data,
                "xmp": self.raw_xmp_data
            } if self.raw_exif_data or self.raw_iptc_data or self.raw_xmp_data else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": str(self.id),
            "file_id": str(self.file_id),
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "lens_model": self.lens_model,
            "focal_length": self.focal_length,
            "aperture": self.aperture,
            "shutter_speed": self.shutter_speed,
            "iso_speed": self.iso_speed,
            "has_gps": self.has_gps,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "date_taken": self.date_taken.isoformat() if self.date_taken else None,
            "is_professional_grade": self.is_professional_grade,
            "equipment_category": self.equipment_category,
            "metadata_completeness_score": self.metadata_completeness_score,
            "extraction_confidence": self.extraction_confidence,
            "metadata_summary": self.metadata_summary,
            "token_count_estimate": self.token_count_estimate,
            "llm_ready": self.llm_ready,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
    }