from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import logging

from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.project import Project
from services.project_service import ProjectService
from services.background_project_deletion_service import background_project_deletion_service
from schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListResponse,
    ProjectQueryParams, ProjectStatusUpdate, ProjectStatsResponse,
    ProjectListWithPagination, ProjectBulkAction, ProjectCreateResponse,
    ProjectUpdateResponse, ProjectDeleteResponse, ProjectBulkActionResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/", response_model=ProjectCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new project"""
    try:
        project_service = ProjectService(db)
        project = await project_service.create_project(project_data, current_user.id)
        
        return ProjectCreateResponse(
            success=True,
            message="Project created successfully",
            project=ProjectResponse.from_orm(project)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project"
        )

@router.get("/", response_model=ProjectListWithPagination)
async def get_projects(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, min_length=1, max_length=255, description="Search term"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("updated_at", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's projects with filtering and pagination"""
    try:
        # Create query parameters
        query_params = ProjectQueryParams(
            page=page,
            limit=limit,
            search=search,
            category=category,
            status=status,
            priority=priority,
            tags=tags,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        project_service = ProjectService(db)
        projects, total = await project_service.get_user_projects(current_user.id, query_params)
        
        # Convert to response models
        project_list = [ProjectListResponse.from_orm(project) for project in projects]
        
        # Calculate pagination info
        total_pages = (total + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1
        
        return ProjectListWithPagination(
            projects=project_list,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting projects: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve projects"
        )

@router.get("/stats", response_model=ProjectStatsResponse)
async def get_project_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's project statistics"""
    try:
        project_service = ProjectService(db)
        stats = await project_service.get_project_stats(current_user.id)
        
        return ProjectStatsResponse(**stats)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve project statistics"
        )

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get project by ID"""
    try:
        project_service = ProjectService(db)
        project = await project_service.get_project_by_id(project_id, current_user.id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return ProjectResponse.from_orm(project)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve project"
        )

@router.put("/{project_id}", response_model=ProjectUpdateResponse)
async def update_project(
    project_id: UUID,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update project"""
    try:
        project_service = ProjectService(db)
        project = await project_service.update_project(project_id, project_data, current_user.id)
        
        return ProjectUpdateResponse(
            success=True,
            message="Project updated successfully",
            project=ProjectResponse.from_orm(project)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project"
        )

@router.patch("/{project_id}/status", response_model=ProjectUpdateResponse)
async def update_project_status(
    project_id: UUID,
    status_data: ProjectStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update project status"""
    try:
        project_service = ProjectService(db)
        project = await project_service.update_project_status(project_id, status_data, current_user.id)
        
        return ProjectUpdateResponse(
            success=True,
            message=f"Project status updated to {status_data.status.value}",
            project=ProjectResponse.from_orm(project)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project status {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project status"
        )

@router.delete("/{project_id}", response_model=ProjectDeleteResponse)
async def delete_project(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete project with comprehensive cleanup (files, folders, data) in background"""
    try:
        project_service = ProjectService(db)
        result = await project_service.delete_project_background(project_id, current_user.id, background_tasks)
        
        return ProjectDeleteResponse(
            success=True,
            message="Project deletion started in background. All files, folders, and data will be permanently removed.",
            project_id=project_id,
            deletion_id=result.get("deletion_id"),
            estimated_time=result.get("estimated_time")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        )

@router.get("/deletion-status/{deletion_id}")
async def get_deletion_status(
    deletion_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get project deletion status"""
    try:
        status = await background_project_deletion_service.get_deletion_status(deletion_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deletion status not found"
            )
        
        # Verify user access
        if status["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return {
            "success": True,
            "deletion_status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deletion status {deletion_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get deletion status"
        )

@router.post("/bulk-action", response_model=ProjectBulkActionResponse)
async def bulk_action_projects(
    bulk_action: ProjectBulkAction,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Perform bulk actions on projects"""
    try:
        project_service = ProjectService(db)
        result = await project_service.bulk_action_projects(bulk_action, current_user.id)
        
        action_messages = {
            "delete": "Projects deleted",
            "archive": "Projects archived",
            "activate": "Projects activated",
            "set_priority": "Project priorities updated"
        }
        
        message = f"{action_messages.get(bulk_action.action, 'Action completed')}. "
        message += f"Processed: {result['processed_count']}, Failed: {result['failed_count']}"
        
        return ProjectBulkActionResponse(
            success=True,
            message=message,
            processed_count=result['processed_count'],
            failed_count=result['failed_count'],
            errors=result.get('errors')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk action: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk action"
        )

# Additional utility endpoints
@router.get("/{project_id}/files/count")
async def get_project_file_count(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get project file count breakdown"""
    try:
        project_service = ProjectService(db)
        project = await project_service.get_project_by_id(project_id, current_user.id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return {
            "total_files": project.file_count,
            "image_count": project.image_count,
            "video_count": project.video_count,
            "document_count": project.document_count,
            "total_size": project.total_file_size,
            "total_size_formatted": project.total_size_formatted
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file count for project {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file count"
        )

@router.post("/{project_id}/duplicate")
async def duplicate_project(
    project_id: UUID,
    new_name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Duplicate a project"""
    try:
        project_service = ProjectService(db)
        original_project = await project_service.get_project_by_id(project_id, current_user.id)
        
        if not original_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Create duplicate project data
        duplicate_name = new_name or f"{original_project.name} (Copy)"
        
        project_data = ProjectCreate(
            name=duplicate_name,
            description=original_project.description,
            category=original_project.category,
            priority=original_project.priority,
            location=original_project.location,
            tags=original_project.tags or [],
            settings=original_project.settings or {},
            metadata=original_project.metadata or {},
            is_public=False  # Always create duplicates as private
        )
        
        duplicate_project = await project_service.create_project(project_data, current_user.id)
        
        return ProjectCreateResponse(
            success=True,
            message="Project duplicated successfully",
            project=ProjectResponse.from_orm(duplicate_project)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating project {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to duplicate project"
        )

@router.post("/{project_id}/recalculate-stats")
async def recalculate_project_statistics(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Manually recalculate project file statistics from actual files in database"""
    try:
        project_service = ProjectService(db)
        
        # Check if project exists and user has access
        project = await project_service.get_project_by_id(project_id, current_user.id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Recalculate statistics
        await project_service.update_project_file_statistics(project_id)
        
        # Get updated project data
        updated_project = await project_service.get_project_by_id(project_id, current_user.id)
        
        return {
            "success": True,
            "message": "Project statistics recalculated successfully",
            "statistics": {
                "file_count": updated_project.file_count,
                "total_file_size": updated_project.total_file_size,
                "total_size_formatted": updated_project.total_size_formatted,
                "image_count": updated_project.image_count,
                "video_count": updated_project.video_count,
                "document_count": updated_project.document_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recalculating statistics for project {project_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to recalculate project statistics"
        ) 