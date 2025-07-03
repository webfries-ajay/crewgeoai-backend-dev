from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import re

class ProjectStatus(str, Enum):
    """Project status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class ProjectCategory(str, Enum):
    """Project category enumeration"""
    MINING = "mining"
    AGRICULTURE = "agriculture"
    CONSTRUCTION = "construction"
    SOLAR = "solar"
    FORESTRY = "forestry"
    URBAN_PLANNING = "urban_planning"
    WIND_MILLS = "wind_mills"

class ProjectPriority(str, Enum):
    """Project priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProjectCollaboratorRole(str, Enum):
    """Project collaborator roles"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    ANALYST = "analyst"

# Base schemas
class ProjectBase(BaseModel):
    """Base project schema"""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=2000, description="Project description")
    category: ProjectCategory = Field(ProjectCategory.AGRICULTURE, description="Project category")
    priority: ProjectPriority = Field(ProjectPriority.MEDIUM, description="Project priority")
    location: Optional[str] = Field(None, max_length=255, description="Project location")
    tags: Optional[List[str]] = Field(default_factory=list, description="Project tags")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project settings")
    project_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project metadata")
    is_public: Optional[bool] = Field(False, description="Is project public")

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Project name cannot be empty')
        # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9\s\-_.,()]+$', v.strip()):
            raise ValueError('Project name contains invalid characters')
        return v.strip()

    @validator('tags')
    def validate_tags(cls, v):
        if v:
            # Validate each tag
            for tag in v:
                if not isinstance(tag, str) or len(tag) > 50:
                    raise ValueError('Each tag must be a string with max 50 characters')
            # Remove duplicates while preserving order
            return list(dict.fromkeys(v))
        return v

    @validator('description')
    def validate_description(cls, v):
        if v:
            return v.strip()
        return v

class ProjectCreate(ProjectBase):
    """Schema for creating a project"""
    pass

class ProjectUpdate(BaseModel):
    """Schema for updating a project"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    category: Optional[ProjectCategory] = None
    priority: Optional[ProjectPriority] = None
    location: Optional[str] = Field(None, max_length=255)
    tags: Optional[List[str]] = None
    settings: Optional[Dict[str, Any]] = None
    project_metadata: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None
    status: Optional[ProjectStatus] = None
    progress: Optional[int] = Field(None, ge=0, le=100)

    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            if not v or not v.strip():
                raise ValueError('Project name cannot be empty')
            if not re.match(r'^[a-zA-Z0-9\s\-_.,()]+$', v.strip()):
                raise ValueError('Project name contains invalid characters')
            return v.strip()
        return v

    @validator('tags')
    def validate_tags(cls, v):
        if v is not None:
            for tag in v:
                if not isinstance(tag, str) or len(tag) > 50:
                    raise ValueError('Each tag must be a string with max 50 characters')
            return list(dict.fromkeys(v))
        return v

    @validator('description')
    def validate_description(cls, v):
        if v is not None:
            return v.strip()
        return v

class ProjectStatusUpdate(BaseModel):
    """Schema for updating project status"""
    status: ProjectStatus
    progress: Optional[int] = Field(None, ge=0, le=100)

class ProjectResponse(BaseModel):
    """Schema for project response"""
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    category: str
    status: str
    progress: int
    priority: str
    location: Optional[str] = None
    settings: Dict[str, Any]
    project_metadata: Dict[str, Any]
    tags: List[str]
    file_count: int
    image_count: int
    video_count: int
    document_count: int
    total_file_size: int
    total_size_formatted: str
    collaborator_count: int
    is_public: bool
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    last_activity_at: Optional[datetime] = None
    last_modified_formatted: str
    days_since_created: int

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

class ProjectListResponse(BaseModel):
    """Schema for project list response"""
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    category: str
    status: str
    progress: int
    priority: str
    location: Optional[str] = None
    file_count: int
    image_count: int
    video_count: int
    total_size_formatted: str
    collaborator_count: int
    is_public: bool
    created_at: datetime
    updated_at: datetime
    last_modified_formatted: str
    days_since_created: int

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

class ProjectStatsResponse(BaseModel):
    """Schema for project statistics"""
    total_projects: int
    active_projects: int
    completed_projects: int
    total_files: int
    total_storage_used: int
    total_storage_formatted: str
    projects_by_category: Dict[str, int]
    projects_by_status: Dict[str, int]
    recent_activity_count: int

# Collaborator schemas
class ProjectCollaboratorBase(BaseModel):
    """Base collaborator schema"""
    user_id: uuid.UUID
    role: ProjectCollaboratorRole = ProjectCollaboratorRole.VIEWER
    permissions: Optional[List[str]] = Field(default_factory=list)

class ProjectCollaboratorCreate(ProjectCollaboratorBase):
    """Schema for adding a collaborator"""
    pass

class ProjectCollaboratorUpdate(BaseModel):
    """Schema for updating a collaborator"""
    role: Optional[ProjectCollaboratorRole] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None

class ProjectCollaboratorResponse(BaseModel):
    """Schema for collaborator response"""
    id: uuid.UUID
    project_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    permissions: List[str]
    invited_by: Optional[uuid.UUID] = None
    invited_at: Optional[datetime] = None
    joined_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# Query schemas
class ProjectQueryParams(BaseModel):
    """Schema for project query parameters"""
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    search: Optional[str] = Field(None, min_length=1, max_length=255, description="Search term")
    category: Optional[ProjectCategory] = Field(None, description="Filter by category")
    status: Optional[ProjectStatus] = Field(None, description="Filter by status")
    priority: Optional[ProjectPriority] = Field(None, description="Filter by priority")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    sort_by: Optional[str] = Field("updated_at", description="Sort field")
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$", description="Sort order")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")

    @validator('search')
    def validate_search(cls, v):
        if v:
            return v.strip()
        return v

    @validator('sort_by')
    def validate_sort_by(cls, v):
        allowed_fields = [
            'name', 'created_at', 'updated_at', 'last_activity_at',
            'status', 'priority', 'category', 'file_count', 'progress'
        ]
        if v not in allowed_fields:
            raise ValueError(f'sort_by must be one of: {", ".join(allowed_fields)}')
        return v

class ProjectListWithPagination(BaseModel):
    """Schema for paginated project list"""
    projects: List[ProjectListResponse]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool

class ProjectBulkAction(BaseModel):
    """Schema for bulk actions on projects"""
    project_ids: List[uuid.UUID] = Field(..., min_items=1, max_items=50)
    action: str = Field(..., pattern="^(delete|archive|activate|set_priority)$")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('project_ids')
    def validate_project_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('Duplicate project IDs are not allowed')
        return v

class ProjectExportRequest(BaseModel):
    """Schema for project export request"""
    project_ids: Optional[List[uuid.UUID]] = None
    format: str = Field("json", pattern="^(json|csv|excel)$")
    include_files: bool = Field(False)
    include_metadata: bool = Field(True)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

# Response schemas for API consistency
class ProjectCreateResponse(BaseModel):
    """Response schema for project creation"""
    success: bool = True
    message: str = "Project created successfully"
    project: ProjectResponse

class ProjectUpdateResponse(BaseModel):
    """Response schema for project update"""
    success: bool = True
    message: str = "Project updated successfully"
    project: ProjectResponse

class ProjectDeleteResponse(BaseModel):
    """Response schema for project deletion"""
    success: bool = True
    message: str = "Project deletion started in background"
    project_id: uuid.UUID
    deletion_id: Optional[str] = None
    estimated_time: Optional[str] = None

class ProjectBulkActionResponse(BaseModel):
    """Response schema for bulk actions"""
    success: bool = True
    message: str
    processed_count: int
    failed_count: int
    errors: Optional[List[Dict[str, Any]]] = None 