from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, BigInteger, ForeignKey, Index, CheckConstraint, Float, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from datetime import datetime
from typing import Optional, Dict, Any, List
import mimetypes
import os

from core.database import Base

class FileType(str, enum.Enum):
    """File type enumeration"""
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    GEOSPATIAL = "geospatial"
    ARCHIVE = "archive"
    OTHER = "other"

class FileStatus(str, enum.Enum):
    """File processing status enumeration"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"

class UploadSessionStatus(str, enum.Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class UploadSession(Base):
    """Track background upload sessions"""
    __tablename__ = "upload_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session info
    session_name = Column(String(255), nullable=False)
    total_files = Column(Integer, nullable=False, default=0)
    total_size = Column(BigInteger, nullable=False, default=0)  # Total size in bytes
    
    # Progress tracking
    status = Column(ENUM(UploadSessionStatus), nullable=False, default=UploadSessionStatus.PENDING)
    uploaded_files = Column(Integer, nullable=False, default=0)
    processed_files = Column(Integer, nullable=False, default=0)
    failed_files = Column(Integer, nullable=False, default=0)
    uploaded_size = Column(BigInteger, nullable=False, default=0)  # Uploaded size in bytes
    
    # Progress percentages
    upload_progress = Column(Float, nullable=False, default=0.0)  # 0-100
    processing_progress = Column(Float, nullable=False, default=0.0)  # 0-100
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    failed_file_details = Column(JSON, nullable=True)  # Store details of failed files
    
    # Relationships
    project = relationship("Project", back_populates="upload_sessions")
    user = relationship("User", back_populates="upload_sessions")
    files = relationship("File", back_populates="upload_session", cascade="all, delete-orphan")

class File(Base):
    __tablename__ = "files"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # File information
    original_filename = Column(Text, nullable=False)  # User's original filename
    stored_filename = Column(Text, nullable=False)  # System filename
    file_path = Column(Text, nullable=False)  # Storage path
    file_type = Column(ENUM(FileType), nullable=False, index=True)  # FileType enum value
    mime_type = Column(String(100), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    
    # File integrity and security
    checksum_md5 = Column(String(32), nullable=True)
    checksum_sha256 = Column(String(64), nullable=True)
    virus_scan_status = Column(String(20), default="pending")  # pending, clean, infected, error
    virus_scan_result = Column(JSONB, default=dict)
    
    # Processing status
    upload_status = Column(ENUM(FileStatus), nullable=False, default=FileStatus.UPLOADING)
    processing_status = Column(String(20), default="pending", index=True)
    is_enabled = Column(Boolean, default=True)  # User toggle for analysis inclusion
    
    # Note: File metadata is now stored in the FileMetadata model via relationship
    
    # Image/Video specific
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration_seconds = Column(Integer, nullable=True)  # For videos
    frame_rate = Column(Integer, nullable=True)  # For videos
    
    # Processing information
    thumbnail_generated = Column(Boolean, default=False)
    preview_generated = Column(Boolean, default=False)
    analysis_completed = Column(Boolean, default=False)
    
    # Upload and ownership
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    upload_session_id = Column(UUID(as_uuid=True), ForeignKey("upload_sessions.id"), nullable=True)
    
    # Tags and organization
    tags = Column(JSONB, default=list)
    description = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="files")
    user = relationship("User", foreign_keys=[user_id], back_populates="files")
    uploader = relationship("User", foreign_keys=[uploaded_by])
    thumbnails = relationship("FileThumbnail", back_populates="file", cascade="all, delete-orphan")
    upload_session = relationship("UploadSession", back_populates="files")
    file_metadata_record = relationship("FileMetadata", back_populates="file", uselist=False, cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="file", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('idx_file_project_type', 'project_id', 'file_type'),
        Index('idx_file_upload_status', 'upload_status'),
        Index('idx_file_processing_status', 'processing_status'),
        Index('idx_file_uploader', 'uploaded_by'),
        Index('idx_file_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<File(id={self.id}, filename={self.original_filename}, type={self.file_type})>"
    
    @hybrid_property
    def file_extension(self) -> str:
        """Get file extension"""
        return os.path.splitext(self.original_filename)[1].lower()
    
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
    def is_image(self) -> bool:
        """Check if file is an image"""
        return self.file_type == FileType.IMAGE.value
    
    @hybrid_property
    def is_video(self) -> bool:
        """Check if file is a video"""
        return self.file_type == FileType.VIDEO.value
    
    @hybrid_property
    def is_document(self) -> bool:
        """Check if file is a document"""
        return self.file_type == FileType.DOCUMENT.value
    
    @hybrid_property
    def is_processing_complete(self) -> bool:
        """Check if processing is complete"""
        return self.processing_status == "processed"
    
    @hybrid_property
    def is_upload_complete(self) -> bool:
        """Check if upload is complete"""
        return self.upload_status == FileStatus.UPLOADED
    
    @hybrid_property
    def has_thumbnails(self) -> bool:
        """Check if file has thumbnails"""
        return self.thumbnail_generated and len(self.thumbnails) > 0
    
    @classmethod
    def determine_file_type(cls, filename: str, mime_type: str = None) -> FileType:
        """Determine file type based on filename and mime type"""
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
        
        if not mime_type:
            return FileType.OTHER
        
        mime_type = mime_type.lower()
        
        if mime_type.startswith('image/'):
            return FileType.IMAGE
        elif mime_type.startswith('video/'):
            return FileType.VIDEO
        elif mime_type in ['application/pdf', 'application/msword', 
                          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                          'text/plain', 'text/csv']:
            return FileType.DOCUMENT
        elif mime_type in ['application/zip', 'application/x-rar-compressed', 
                          'application/x-tar', 'application/gzip']:
            return FileType.ARCHIVE
        elif 'geojson' in mime_type or 'kml' in mime_type or 'shapefile' in mime_type:
            return FileType.GEOSPATIAL
        else:
            return FileType.OTHER
    
    def update_processing_status(self, status: str) -> None:
        """Update processing status"""
        self.processing_status = status
        
        if status == "processed":
            self.processed_at = datetime.utcnow()
            self.analysis_completed = True
        
        self.updated_at = datetime.utcnow()
    
    def update_upload_status(self, status: FileStatus) -> None:
        """Update upload status"""
        self.upload_status = status
        self.updated_at = datetime.utcnow()
    
    def mark_accessed(self) -> None:
        """Mark file as accessed"""
        self.last_accessed_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the file"""
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the file"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert file to dictionary representation"""
        data = {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "stored_filename": self.stored_filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "file_size_formatted": self.file_size_formatted,
            "file_size_mb": self.file_size_mb,
            "upload_status": self.upload_status,
            "processing_status": self.processing_status,
            "is_enabled": self.is_enabled,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "frame_rate": self.frame_rate,
            "thumbnail_generated": self.thumbnail_generated,
            "preview_generated": self.preview_generated,
            "analysis_completed": self.analysis_completed,
            "tags": self.tags,
            "description": self.description,
            "uploaded_by": str(self.uploaded_by),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
        }
        
        if include_metadata:
            data.update({
                "checksum_md5": self.checksum_md5,
                "checksum_sha256": self.checksum_sha256,
                "virus_scan_status": self.virus_scan_status,
                "metadata": self.file_metadata_record.to_dict() if self.file_metadata_record else None,
            })
        
        return data

class FileThumbnail(Base):
    __tablename__ = "file_thumbnails"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Thumbnail information
    thumbnail_type = Column(String(20), nullable=False)  # 'small', 'medium', 'large'
    thumbnail_path = Column(Text, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    file = relationship("File", back_populates="thumbnails")
    
    # Constraints
    __table_args__ = (
        Index('idx_file_thumbnail_file_type', 'file_id', 'thumbnail_type'),
    )
    
    def __repr__(self):
        return f"<FileThumbnail(file_id={self.file_id}, type={self.thumbnail_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thumbnail to dictionary representation"""
        return {
            "id": str(self.id),
            "file_id": str(self.file_id),
            "thumbnail_type": self.thumbnail_type,
            "thumbnail_path": self.thumbnail_path,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        } 