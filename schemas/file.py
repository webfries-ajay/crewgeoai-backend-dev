from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID

class FileBase(BaseModel):
    """Base file schema"""
    original_filename: str = Field(..., description="Original filename")
    description: Optional[str] = Field(None, description="File description")
    tags: List[str] = Field(default_factory=list, description="File tags")

class FileCreate(FileBase):
    """Schema for creating a file"""
    pass

class FileUpdate(BaseModel):
    """Schema for updating a file"""
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_enabled: Optional[bool] = None

class FileThumbnailResponse(BaseModel):
    """Schema for file thumbnail response"""
    id: UUID
    thumbnail_type: str
    width: int
    height: int
    file_size: int
    created_at: datetime

    class Config:
        from_attributes = True

class FileResponse(BaseModel):
    """Schema for file response"""
    id: UUID
    project_id: UUID
    filename: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size: int
    file_size_formatted: str
    
    # Status fields
    upload_status: str
    processing_status: str
    is_enabled: bool
    
    # Metadata fields
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[int] = None
    frame_rate: Optional[int] = None
    
    # Processing flags
    thumbnail_generated: bool
    preview_generated: bool
    analysis_completed: bool
    
    # Note: File metadata is now available through the metadata relationship
    
    # Organization
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    # Relationships
    thumbnails: List[FileThumbnailResponse] = Field(default_factory=list)
    
    # Uploader info
    uploaded_by: UUID

    class Config:
        from_attributes = True

class FileListItem(BaseModel):
    """Schema for file list item (lighter version)"""
    id: UUID
    original_filename: str
    file_type: str
    file_size: int
    file_size_formatted: str
    upload_status: str
    processing_status: str
    thumbnail_generated: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class FileListResponse(BaseModel):
    """Schema for file list response"""
    files: List[FileListItem]
    total: int
    skip: int
    limit: int

class FileUploadItem(BaseModel):
    """Schema for uploaded file item"""
    id: str
    filename: str
    size: int
    type: str
    status: str

class FileUploadError(BaseModel):
    """Schema for file upload error"""
    filename: str
    error: str

class FileUploadResponse(BaseModel):
    """Schema for file upload response"""
    success: bool
    message: str
    uploaded_files: List[FileUploadItem]
    failed_files: List[FileUploadError]
    total_uploaded: int
    total_failed: int

class FileProcessingStatus(BaseModel):
    """Schema for file processing status"""
    file_id: UUID
    filename: str
    upload_status: str
    processing_status: str
    progress_percentage: Optional[int] = None
    error_message: Optional[str] = None

class ProcessingStatusResponse(BaseModel):
    """Schema for processing status response"""
    project_id: UUID
    total_files: int
    completed_files: int
    processing_files: int
    failed_files: int
    pending_files: int
    files: List[FileProcessingStatus]

class BulkDeleteRequest(BaseModel):
    """Schema for bulk delete request"""
    file_ids: List[str] = Field(..., min_items=1, max_items=100)

class BulkDeleteResponse(BaseModel):
    """Schema for bulk delete response"""
    success: bool
    message: str
    deleted_count: int
    failed_count: int
    errors: List[str] = Field(default_factory=list)

class UploadSessionResponse(BaseModel):
    """Schema for background upload session response"""
    success: bool
    message: str
    session_id: str
    session_name: str
    total_files: int
    total_size: int
    status: str

# File type validation
ALLOWED_FILE_TYPES = {
    "image", "video", "document", "geospatial", "archive", "other"
}

# File status validation
ALLOWED_UPLOAD_STATUSES = {
    "uploading", "pending", "processing", "completed", "failed", "quarantined"
}

ALLOWED_PROCESSING_STATUSES = {
    "pending", "processing", "completed", "failed"
} 