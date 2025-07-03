from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, text, update
from sqlalchemy.orm import selectinload, joinedload
from fastapi import HTTPException, status
import logging

from models.project import Project, ProjectCollaborator, ProjectStatus, ProjectCategory, ProjectPriority, ProjectCollaboratorRole
from models.user import User
from models.file import File
from schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectQueryParams, ProjectStatusUpdate,
    ProjectCollaboratorCreate, ProjectCollaboratorUpdate, ProjectBulkAction
)

logger = logging.getLogger(__name__)

class ProjectService:
    """Service class for project operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_project(self, project_data: ProjectCreate, owner_id: UUID) -> Project:
        """Create a new project"""
        try:
            # Check if user exists
            user_query = select(User).where(User.id == owner_id)
            user_result = await self.db.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Create project
            project = Project(
                name=project_data.name,
                description=project_data.description,
                category=project_data.category.value,
                priority=project_data.priority.value,
                location=project_data.location,
                tags=project_data.tags or [],
                settings=project_data.settings or {},
                project_metadata=project_data.project_metadata or {},
                is_public=project_data.is_public or False,
                owner_id=owner_id,
                status=ProjectStatus.DRAFT.value
            )
            
            self.db.add(project)
            await self.db.commit()
            await self.db.refresh(project)
            
            logger.info(f"Project created: {project.id} by user {owner_id}")
            return project
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create project"
            )
    
    async def get_project_by_id(self, project_id: UUID, user_id: UUID) -> Optional[Project]:
        """Get project by ID with access control"""
        try:
            query = (
                select(Project)
                .options(
                    selectinload(Project.collaborators),
                    selectinload(Project.owner)
                )
                .where(Project.id == project_id)
            )
            
            result = await self.db.execute(query)
            project = result.scalar_one_or_none()
            
            if not project:
                return None
            
            # Check access permissions
            if not await self._has_project_access(project, user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this project"
                )
            
            return project
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting project {project_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve project"
            )
    
    async def get_user_projects(
        self, 
        user_id: UUID, 
        query_params: ProjectQueryParams
    ) -> Tuple[List[Project], int]:
        """Get user's projects with filtering and pagination"""
        try:
            # Base query
            base_query = (
                select(Project)
                .options(selectinload(Project.owner))
                .where(
                    or_(
                        Project.owner_id == user_id,
                        Project.id.in_(
                            select(ProjectCollaborator.project_id)
                            .where(
                                and_(
                                    ProjectCollaborator.user_id == user_id,
                                    ProjectCollaborator.is_active == True
                                )
                            )
                        )
                    )
                )
            )
            
            # Apply filters
            if query_params.search:
                search_term = f"%{query_params.search}%"
                base_query = base_query.where(
                    or_(
                        Project.name.ilike(search_term),
                        Project.description.ilike(search_term),
                        Project.location.ilike(search_term)
                    )
                )
            
            if query_params.category:
                base_query = base_query.where(Project.category == query_params.category.value)
            
            if query_params.status:
                base_query = base_query.where(Project.status == query_params.status.value)
            
            if query_params.priority:
                base_query = base_query.where(Project.priority == query_params.priority.value)
            
            if query_params.tags:
                for tag in query_params.tags:
                    base_query = base_query.where(Project.tags.contains([tag]))
            
            if query_params.date_from:
                base_query = base_query.where(Project.created_at >= query_params.date_from)
            
            if query_params.date_to:
                base_query = base_query.where(Project.created_at <= query_params.date_to)
            
            # Count total
            count_query = select(func.count()).select_from(base_query.subquery())
            count_result = await self.db.execute(count_query)
            total = count_result.scalar()
            
            # Apply sorting
            sort_column = getattr(Project, query_params.sort_by, Project.updated_at)
            if query_params.sort_order == "asc":
                base_query = base_query.order_by(asc(sort_column))
            else:
                base_query = base_query.order_by(desc(sort_column))
            
            # Apply pagination
            offset = (query_params.page - 1) * query_params.limit
            base_query = base_query.offset(offset).limit(query_params.limit)
            
            # Execute query
            result = await self.db.execute(base_query)
            projects = result.scalars().all()
            
            return list(projects), total
            
        except Exception as e:
            logger.error(f"Error getting user projects: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve projects"
            )
    
    async def update_project(
        self, 
        project_id: UUID, 
        project_data: ProjectUpdate, 
        user_id: UUID
    ) -> Project:
        """Update project"""
        try:
            project = await self.get_project_by_id(project_id, user_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            # Check if user has edit permissions
            if not await self._has_project_edit_access(project, user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions to edit this project"
                )
            
            # Update fields
            update_data = project_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(project, field):
                    if field in ['category', 'priority', 'status'] and value:
                        setattr(project, field, value.value if hasattr(value, 'value') else value)
                    else:
                        setattr(project, field, value)
            
            project.update_activity()
            
            await self.db.commit()
            await self.db.refresh(project)
            
            logger.info(f"Project updated: {project.id} by user {user_id}")
            return project
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating project {project_id}: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update project"
            )
    
    async def update_project_status(
        self, 
        project_id: UUID, 
        status_data: ProjectStatusUpdate, 
        user_id: UUID
    ) -> Project:
        """Update project status"""
        try:
            project = await self.get_project_by_id(project_id, user_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            # Check permissions
            if not await self._has_project_edit_access(project, user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions to update project status"
                )
            
            # Update status
            project.set_status(ProjectStatus(status_data.status.value))
            
            if status_data.progress is not None:
                project.progress = status_data.progress
            
            await self.db.commit()
            await self.db.refresh(project)
            
            logger.info(f"Project status updated: {project.id} to {status_data.status.value}")
            return project
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating project status: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update project status"
            )
    
    async def delete_project(self, project_id: UUID, user_id: UUID) -> bool:
        """Delete project (only owner can delete) - DEPRECATED: Use delete_project_background instead"""
        try:
            project = await self.get_project_by_id(project_id, user_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            # Only owner can delete
            if project.owner_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only project owner can delete the project"
                )
            
            await self.db.delete(project)
            await self.db.commit()
            
            logger.info(f"Project deleted: {project_id} by user {user_id}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete project"
            )
    
    async def delete_project_background(
        self, 
        project_id: UUID, 
        user_id: UUID, 
        background_tasks
    ) -> Dict[str, Any]:
        """Start background project deletion with comprehensive cleanup"""
        try:
            # Verify project exists and user has permission
            project = await self.get_project_by_id(project_id, user_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )
            
            # Only owner can delete
            if project.owner_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only project owner can delete the project"
                )
            
            # Import here to avoid circular imports
            from services.background_project_deletion_service import background_project_deletion_service
            
            # Start background deletion
            result = await background_project_deletion_service.start_project_deletion(
                str(project_id),
                str(user_id),
                background_tasks
            )
            
            logger.info(f"Background deletion started for project {project_id} by user {user_id}")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting project deletion {project_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start project deletion"
            )
    
    async def get_project_stats(self, user_id: UUID) -> Dict[str, Any]:
        """Get user's project statistics"""
        try:
            # Get user's projects
            user_projects_query = (
                select(Project)
                .where(
                    or_(
                        Project.owner_id == user_id,
                        Project.id.in_(
                            select(ProjectCollaborator.project_id)
                            .where(
                                and_(
                                    ProjectCollaborator.user_id == user_id,
                                    ProjectCollaborator.is_active == True
                                )
                            )
                        )
                    )
                )
            )
            
            result = await self.db.execute(user_projects_query)
            projects = result.scalars().all()
            
            # Calculate statistics
            total_projects = len(projects)
            active_projects = sum(1 for p in projects if p.status == ProjectStatus.ACTIVE.value)
            completed_projects = sum(1 for p in projects if p.status == ProjectStatus.COMPLETED.value)
            total_files = sum(p.file_count for p in projects)
            total_storage = sum(p.total_file_size for p in projects)
            
            # Projects by category
            projects_by_category = {}
            for project in projects:
                category = project.category
                projects_by_category[category] = projects_by_category.get(category, 0) + 1
            
            # Projects by status
            projects_by_status = {}
            for project in projects:
                status = project.status
                projects_by_status[status] = projects_by_status.get(status, 0) + 1
            
            # Recent activity (last 7 days)
            recent_date = datetime.utcnow() - timedelta(days=7)
            recent_activity = sum(
                1 for p in projects 
                if p.last_activity_at and p.last_activity_at.replace(tzinfo=None) > recent_date
            )
            
            # Format total storage
            total_storage_formatted = self._format_file_size(total_storage)
            
            return {
                "total_projects": total_projects,
                "active_projects": active_projects,
                "completed_projects": completed_projects,
                "total_files": total_files,
                "total_storage_used": total_storage,
                "total_storage_formatted": total_storage_formatted,
                "projects_by_category": projects_by_category,
                "projects_by_status": projects_by_status,
                "recent_activity_count": recent_activity
            }
            
        except Exception as e:
            logger.error(f"Error getting project stats: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve project statistics"
            )
    
    async def bulk_action_projects(
        self, 
        bulk_action: ProjectBulkAction, 
        user_id: UUID
    ) -> Dict[str, Any]:
        """Perform bulk actions on projects"""
        try:
            processed_count = 0
            failed_count = 0
            errors = []
            
            for project_id in bulk_action.project_ids:
                try:
                    project = await self.get_project_by_id(project_id, user_id)
                    if not project:
                        errors.append({
                            "project_id": str(project_id),
                            "error": "Project not found"
                        })
                        failed_count += 1
                        continue
                    
                    # Check permissions based on action
                    if bulk_action.action == "delete":
                        if project.owner_id != user_id:
                            errors.append({
                                "project_id": str(project_id),
                                "error": "Only owner can delete project"
                            })
                            failed_count += 1
                            continue
                        await self.db.delete(project)
                    
                    elif bulk_action.action == "archive":
                        if not await self._has_project_edit_access(project, user_id):
                            errors.append({
                                "project_id": str(project_id),
                                "error": "Insufficient permissions"
                            })
                            failed_count += 1
                            continue
                        project.set_status(ProjectStatus.ARCHIVED)
                    
                    elif bulk_action.action == "activate":
                        if not await self._has_project_edit_access(project, user_id):
                            errors.append({
                                "project_id": str(project_id),
                                "error": "Insufficient permissions"
                            })
                            failed_count += 1
                            continue
                        project.set_status(ProjectStatus.ACTIVE)
                    
                    elif bulk_action.action == "set_priority":
                        if not await self._has_project_edit_access(project, user_id):
                            errors.append({
                                "project_id": str(project_id),
                                "error": "Insufficient permissions"
                            })
                            failed_count += 1
                            continue
                        
                        priority = bulk_action.parameters.get("priority")
                        if priority in [p.value for p in ProjectPriority]:
                            project.priority = priority
                            project.update_activity()
                        else:
                            errors.append({
                                "project_id": str(project_id),
                                "error": "Invalid priority value"
                            })
                            failed_count += 1
                            continue
                    
                    processed_count += 1
                    
                except Exception as e:
                    errors.append({
                        "project_id": str(project_id),
                        "error": str(e)
                    })
                    failed_count += 1
            
            await self.db.commit()
            
            return {
                "processed_count": processed_count,
                "failed_count": failed_count,
                "errors": errors if errors else None
            }
            
        except Exception as e:
            logger.error(f"Error in bulk action: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to perform bulk action"
            )
    
    # Helper methods
    async def _has_project_access(self, project: Project, user_id: UUID) -> bool:
        """Check if user has access to project"""
        # Owner has access
        if project.owner_id == user_id:
            return True
        
        # Public projects are accessible
        if project.is_public:
            return True
        
        # Check collaboration
        collab_query = (
            select(ProjectCollaborator)
            .where(
                and_(
                    ProjectCollaborator.project_id == project.id,
                    ProjectCollaborator.user_id == user_id,
                    ProjectCollaborator.is_active == True
                )
            )
        )
        result = await self.db.execute(collab_query)
        collaborator = result.scalar_one_or_none()
        
        return collaborator is not None
    
    async def _has_project_edit_access(self, project: Project, user_id: UUID) -> bool:
        """Check if user has edit access to project"""
        # Owner has edit access
        if project.owner_id == user_id:
            return True
        
        # Check collaboration with edit permissions
        collab_query = (
            select(ProjectCollaborator)
            .where(
                and_(
                    ProjectCollaborator.project_id == project.id,
                    ProjectCollaborator.user_id == user_id,
                    ProjectCollaborator.is_active == True,
                    ProjectCollaborator.role.in_(["owner", "editor"])
                )
            )
        )
        result = await self.db.execute(collab_query)
        collaborator = result.scalar_one_or_none()
        
        return collaborator is not None
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if not size_bytes:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB" 
    
    async def update_project_file_statistics(self, project_id: UUID) -> None:
        """Recalculate and update project file statistics from actual files in database"""
        try:
            # Get all files for the project
            files_query = select(File).where(File.project_id == project_id)
            result = await self.db.execute(files_query)
            files = result.scalars().all()
            
            # Calculate statistics
            total_files = len(files)
            total_size = sum(file.file_size for file in files if file.file_size)
            image_count = sum(1 for file in files if file.file_type == 'image')
            video_count = sum(1 for file in files if file.file_type == 'video')
            document_count = sum(1 for file in files if file.file_type == 'document')
            
            # Update project with calculated statistics
            update_query = (
                update(Project)
                .where(Project.id == project_id)
                .values(
                    file_count=total_files,
                    total_file_size=total_size,
                    image_count=image_count,
                    video_count=video_count,
                    document_count=document_count,
                    updated_at=func.now()
                )
            )
            
            await self.db.execute(update_query)
            await self.db.commit()
            
            logger.info(f"Updated project {project_id} statistics: {total_files} files, {total_size} bytes")
            
        except Exception as e:
            logger.error(f"Error updating project file statistics for {project_id}: {str(e)}")
            await self.db.rollback()
            raise 