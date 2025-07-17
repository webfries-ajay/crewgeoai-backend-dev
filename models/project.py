from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, BigInteger, ForeignKey, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
# from geoalchemy2 import Geometry
import uuid
import enum
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.database import Base

class ProjectStatus(enum.Enum):
    """Project status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class ProjectCategory(enum.Enum):
    """Project category enumeration"""
    MINING = "mining"
    AGRICULTURE = "agriculture"
    CONSTRUCTION = "construction"
    SOLAR = "solar"
    FORESTRY = "forestry"
    URBAN_PLANNING = "urban_planning"
    WIND_MILLS = "wind_mills"

class ProjectPriority(enum.Enum):
    """Project priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Project(Base):
    __tablename__ = "projects"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False, default=ProjectCategory.AGRICULTURE.value)
    
    # Project status and progress
    status = Column(String(20), default=ProjectStatus.DRAFT.value, index=True)
    progress = Column(Integer, default=0)
    priority = Column(String(20), default=ProjectPriority.MEDIUM.value)
    
    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Project configuration and metadata
    settings = Column(JSONB, default=dict)  # User-configurable project settings
    project_metadata = Column(JSONB, default=dict)  # Project-specific metadata
    tags = Column(JSONB, default=list)  # Project tags for organization
    
    # Geographic information
    location = Column(String(255), nullable=True)  # Human-readable location
    # location_bounds = Column(Geometry('POLYGON', srid=4326), nullable=True)  # Project area boundaries
    
    # File statistics (updated by triggers/functions)
    total_file_size = Column(BigInteger, default=0)
    file_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)
    document_count = Column(Integer, default=0)
    
    # Processing information
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion_at = Column(DateTime(timezone=True), nullable=True)
    
    # Collaboration
    is_public = Column(Boolean, default=False)
    collaborator_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now())
    archived_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    collaborators = relationship("ProjectCollaborator", back_populates="project", cascade="all, delete-orphan")
    files = relationship("File", back_populates="project", cascade="all, delete-orphan")
    upload_sessions = relationship("UploadSession", back_populates="project", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('progress >= 0 AND progress <= 100', name='check_progress_range'),
        Index('idx_project_owner_status', 'owner_id', 'status'),
        Index('idx_project_category', 'category'),
        Index('idx_project_created_at', 'created_at'),
        Index('idx_project_updated_at', 'updated_at'),
        # Index('idx_project_location_bounds', 'location_bounds', postgresql_using='gist'),
    )
    
    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name}, status={self.status})>"
    
    @hybrid_property
    def full_name(self) -> str:
        """Get project full name with category"""
        return f"{self.name} ({self.category})"
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if project is in active status"""
        return self.status == ProjectStatus.ACTIVE.value
    
    @hybrid_property
    def is_completed(self) -> bool:
        """Check if project is completed"""
        return self.status == ProjectStatus.COMPLETED.value
    
    @hybrid_property
    def is_processing(self) -> bool:
        """Check if project is currently processing"""
        return self.status == ProjectStatus.PROCESSING.value
    
    @hybrid_property
    def total_size_mb(self) -> float:
        """Get total file size in MB"""
        return round(self.total_file_size / (1024 * 1024), 2) if self.total_file_size else 0
    
    @hybrid_property
    def total_size_gb(self) -> float:
        """Get total file size in GB"""
        return round(self.total_file_size / (1024 * 1024 * 1024), 2) if self.total_file_size else 0
    
    @hybrid_property
    def total_size_formatted(self) -> str:
        """Get formatted file size"""
        if not self.total_file_size:
            return "0 B"
        
        size = self.total_file_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    @hybrid_property
    def days_since_created(self) -> int:
        """Get days since project creation"""
        if self.created_at:
            return (datetime.utcnow() - self.created_at.replace(tzinfo=None)).days
        return 0
    
    @hybrid_property
    def last_modified_formatted(self) -> str:
        """Get formatted last modified time"""
        if not self.updated_at:
            return "Never"
        
        now = datetime.utcnow()
        diff = now - self.updated_at.replace(tzinfo=None)
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()
    
    def update_file_stats(self, file_count: int = None, image_count: int = None, 
                         video_count: int = None, document_count: int = None, 
                         total_size: int = None) -> None:
        """Update file statistics"""
        if file_count is not None:
            self.file_count = file_count
        if image_count is not None:
            self.image_count = image_count
        if video_count is not None:
            self.video_count = video_count
        if document_count is not None:
            self.document_count = document_count
        if total_size is not None:
            self.total_file_size = total_size
        
        self.update_activity()
    
    def set_status(self, status: ProjectStatus) -> None:
        """Set project status with appropriate timestamps"""
        self.status = status.value
        
        if status == ProjectStatus.PROCESSING:
            self.processing_started_at = datetime.utcnow()
            self.processing_completed_at = None
        elif status == ProjectStatus.COMPLETED:
            self.processing_completed_at = datetime.utcnow()
        elif status == ProjectStatus.ARCHIVED:
            self.archived_at = datetime.utcnow()
        
        self.update_activity()
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert project to dictionary representation"""
        data = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "progress": self.progress,
            "priority": self.priority,
            "location": self.location,
            "settings": self.settings,
            "metadata": self.project_metadata,
            "tags": self.tags,
            "file_count": self.file_count,
            "image_count": self.image_count,
            "video_count": self.video_count,
            "document_count": self.document_count,
            "total_file_size": self.total_file_size,
            "total_size_formatted": self.total_size_formatted,
            "collaborator_count": self.collaborator_count,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
            "last_modified_formatted": self.last_modified_formatted,
            "days_since_created": self.days_since_created,
            "owner_id": str(self.owner_id)
        }
        
        if include_sensitive:
            data.update({
                "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
                "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
                "estimated_completion_at": self.estimated_completion_at.isoformat() if self.estimated_completion_at else None,
                "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            })
        
        return data

class ProjectCollaboratorRole(enum.Enum):
    """Project collaborator roles"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    ANALYST = "analyst"

class ProjectCollaborator(Base):
    __tablename__ = "project_collaborators"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Collaboration details
    role = Column(String(20), nullable=False, default=ProjectCollaboratorRole.VIEWER.value)
    permissions = Column(JSONB, default=list)  # Additional permissions
    
    # Invitation details
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    invited_at = Column(DateTime(timezone=True), server_default=func.now())
    joined_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="collaborators")
    user = relationship("User", foreign_keys=[user_id])
    inviter = relationship("User", foreign_keys=[invited_by])
    
    # Constraints
    __table_args__ = (
        Index('idx_project_collaborator_unique', 'project_id', 'user_id', unique=True),
        Index('idx_project_collaborator_project', 'project_id'),
        Index('idx_project_collaborator_user', 'user_id'),
    )
    
    def __repr__(self):
        return f"<ProjectCollaborator(project_id={self.project_id}, user_id={self.user_id}, role={self.role})>"
    
    def accept_invitation(self) -> None:
        """Accept collaboration invitation"""
        self.joined_at = datetime.utcnow()
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collaborator to dictionary representation"""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "user_id": str(self.user_id),
            "role": self.role,
            "permissions": self.permissions,
            "invited_by": str(self.invited_by) if self.invited_by else None,
            "invited_at": self.invited_at.isoformat() if self.invited_at else None,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        } 