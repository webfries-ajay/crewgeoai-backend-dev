from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, BigInteger, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import uuid
import enum
from datetime import datetime
from typing import Optional, Dict, Any

from core.database import Base

class ProcessedImageType(str, enum.Enum):
    """Processed image type enumeration"""
    NDMI = "ndmi"
    NDVI = "ndvi"

class ProcessedImage(Base):
    """Model for processed images (NDMI, NDVI, etc.)"""
    __tablename__ = "processed_images"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Processed image information
    processed_image_type = Column(ENUM(ProcessedImageType), nullable=False, index=True)
    processed_image_path = Column(Text, nullable=False)  # Storage path
    original_filename = Column(Text, nullable=False)  # Original file name
    processed_filename = Column(Text, nullable=False)  # Processed file name
    
    # Image dimensions and size
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(BigInteger, nullable=False)
    
    # Processing statistics and metadata
    processing_stats = Column(JSONB, default=dict)  # Processing statistics from NDMI/NDVI services
    
    # Processing status
    processing_status = Column(String(20), default="completed")  # completed, failed, processing
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    file = relationship("File")
    
    # Constraints
    __table_args__ = (
        Index('idx_processed_image_file_type', 'file_id', 'processed_image_type'),
        Index('idx_processed_image_type', 'processed_image_type'),
        Index('idx_processed_image_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ProcessedImage(id={self.id}, type={self.processed_image_type}, file_id={self.file_id})>"
    
    @hybrid_property
    def file_size_formatted(self) -> str:
        """Get formatted file size"""
        if not self.file_size:
            return "0 B"
        
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    @hybrid_property
    def file_size_mb(self) -> float:
        """Get file size in MB"""
        return round(self.file_size / (1024 * 1024), 2) if self.file_size else 0
    
    @hybrid_property
    def is_ndmi(self) -> bool:
        """Check if processed image is NDMI"""
        return self.processed_image_type == ProcessedImageType.NDMI
    
    @hybrid_property
    def is_ndvi(self) -> bool:
        """Check if processed image is NDVI"""
        return self.processed_image_type == ProcessedImageType.NDVI
    
    @hybrid_property
    def is_processing_complete(self) -> bool:
        """Check if processing is complete"""
        return self.processing_status == "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "file_id": str(self.file_id),
            "processed_image_type": self.processed_image_type.value,
            "processed_image_path": self.processed_image_path,
            "original_filename": self.original_filename,
            "processed_filename": self.processed_filename,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "file_size_formatted": self.file_size_formatted,
            "processing_stats": self.processing_stats,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        } 