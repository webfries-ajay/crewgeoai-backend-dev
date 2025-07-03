import asyncio
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from core.database import get_db
from models.project import Project, ProjectCollaborator
from models.file import File, UploadSession, FileThumbnail
from models.file_metadata import FileMetadata
from services.file_service import FileService

logger = logging.getLogger(__name__)

class BackgroundProjectDeletionService:
    """Service for handling comprehensive project deletion in the background"""
    
    def __init__(self):
        self.active_deletions = {}  # Track active deletion processes
        self.file_service = FileService()
    
    async def start_project_deletion(
        self,
        project_id: str,
        user_id: str,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Start background project deletion process"""
        
        deletion_id = f"del_{project_id}"
        
        # Track deletion process
        self.active_deletions[deletion_id] = {
            "project_id": project_id,
            "user_id": user_id,
            "status": "starting",
            "started_at": datetime.now(),
            "progress": 0,
            "current_step": "Initializing deletion process",
            "total_steps": 6,
            "completed_steps": 0,
            "errors": []
        }
        
        # Start background task
        background_tasks.add_task(
            self._process_project_deletion,
            project_id,
            user_id,
            deletion_id
        )
        
        logger.info(f"Started background deletion for project {project_id}")
        
        return {
            "success": True,
            "deletion_id": deletion_id,
            "message": "Project deletion started in background",
            "estimated_time": "2-5 minutes depending on project size"
        }
    
    async def _process_project_deletion(
        self,
        project_id: str,
        user_id: str,
        deletion_id: str
    ):
        """Process complete project deletion in background"""
        
        try:
            async for db in get_db():
                # Update status
                self._update_deletion_status(
                    deletion_id, 
                    "in_progress", 
                    10, 
                    "Loading project data"
                )
                
                # Step 1: Load project with all relationships
                logger.info(f"Step 1: Loading project {project_id}")
                project = await self._load_project_with_relationships(db, project_id, user_id)
                
                if not project:
                    self._update_deletion_status(
                        deletion_id, 
                        "failed", 
                        0, 
                        "Project not found or access denied"
                    )
                    return
                
                # Step 2: Delete all project files from storage
                self._update_deletion_status(
                    deletion_id, 
                    "in_progress", 
                    25, 
                    f"Deleting {project.file_count} files from storage"
                )
                logger.info(f"Step 2: Deleting files for project {project_id}")
                await self._delete_project_files(db, project, deletion_id)
                
                # Step 3: Delete project folders
                self._update_deletion_status(
                    deletion_id, 
                    "in_progress", 
                    50, 
                    "Removing project directories"
                )
                logger.info(f"Step 3: Deleting folders for project {project_id}")
                await self._delete_project_folders(project, user_id)
                
                # Step 4: Delete upload sessions
                self._update_deletion_status(
                    deletion_id, 
                    "in_progress", 
                    65, 
                    "Cleaning up upload sessions"
                )
                logger.info(f"Step 4: Deleting upload sessions for project {project_id}")
                await self._delete_upload_sessions(db, project_id)
                
                # Step 5: Delete collaborators
                self._update_deletion_status(
                    deletion_id, 
                    "in_progress", 
                    80, 
                    "Removing project collaborators"
                )
                logger.info(f"Step 5: Deleting collaborators for project {project_id}")
                await self._delete_project_collaborators(db, project_id)
                
                # Step 6: Delete project record
                self._update_deletion_status(
                    deletion_id, 
                    "in_progress", 
                    95, 
                    "Removing project from database"
                )
                logger.info(f"Step 6: Deleting project record {project_id}")
                await self._delete_project_record(db, project)
                
                # Complete
                self._update_deletion_status(
                    deletion_id, 
                    "completed", 
                    100, 
                    "Project deletion completed successfully"
                )
                
                logger.info(f"Project {project_id} deletion completed successfully")
                
        except Exception as e:
            logger.error(f"Error in project deletion {project_id}: {e}")
            self._update_deletion_status(
                deletion_id, 
                "failed", 
                0, 
                f"Deletion failed: {str(e)}"
            )
    
    async def _load_project_with_relationships(
        self, 
        db: AsyncSession, 
        project_id: str, 
        user_id: str
    ) -> Optional[Project]:
        """Load project with all relationships"""
        
        try:
            # Load project with relationships
            stmt = (
                select(Project)
                .options(
                    selectinload(Project.files),
                    selectinload(Project.upload_sessions),
                    selectinload(Project.collaborators)
                )
                .where(Project.id == project_id)
            )
            
            result = await db.execute(stmt)
            project = result.scalar_one_or_none()
            
            if not project:
                return None
            
            # Check if user is owner
            if str(project.owner_id) != user_id:
                logger.warning(f"User {user_id} attempted to delete project {project_id} without ownership")
                return None
            
            return project
            
        except Exception as e:
            logger.error(f"Error loading project {project_id}: {e}")
            return None
    
    async def _delete_project_files(
        self, 
        db: AsyncSession, 
        project: Project, 
        deletion_id: str
    ):
        """Delete all project files from storage and database"""
        
        try:
            files_deleted = 0
            files_failed = 0
            
            for file_record in project.files:
                try:
                    # Delete physical file
                    if file_record.file_path:
                        file_path = Path(file_record.file_path)
                        if file_path.exists():
                            file_path.unlink()
                            logger.debug(f"Deleted file: {file_path}")
                    
                    # Delete thumbnails
                    await self._delete_file_thumbnails(db, file_record.id)
                    
                    # Delete file metadata records
                    await self._delete_file_metadata(db, file_record.id)
                    
                    # Delete file record from database
                    await db.delete(file_record)
                    files_deleted += 1
                    
                    # Update progress periodically
                    if files_deleted % 10 == 0:
                        progress = 25 + (files_deleted / len(project.files)) * 20
                        self._update_deletion_status(
                            deletion_id,
                            "in_progress",
                            progress,
                            f"Deleted {files_deleted}/{len(project.files)} files"
                        )
                    
                except Exception as e:
                    logger.error(f"Error deleting file {file_record.id}: {e}")
                    files_failed += 1
            
            await db.commit()
            logger.info(f"Deleted {files_deleted} files, {files_failed} failed for project {project.id}")
            
        except Exception as e:
            logger.error(f"Error deleting project files: {e}")
            raise
    
    async def _delete_file_thumbnails(self, db: AsyncSession, file_id: UUID):
        """Delete all thumbnails for a file"""
        try:
            # Delete thumbnail records
            stmt = delete(FileThumbnail).where(FileThumbnail.file_id == file_id)
            await db.execute(stmt)
            
        except Exception as e:
            logger.error(f"Error deleting thumbnails for file {file_id}: {e}")
    
    async def _delete_file_metadata(self, db: AsyncSession, file_id: UUID):
        """Delete all metadata records for a file"""
        try:
            # Delete file metadata records
            stmt = delete(FileMetadata).where(FileMetadata.file_id == file_id)
            result = await db.execute(stmt)
            
            if result.rowcount > 0:
                logger.debug(f"Deleted {result.rowcount} metadata records for file {file_id}")
            
        except Exception as e:
            logger.error(f"Error deleting metadata for file {file_id}: {e}")
    
    async def _delete_project_folders(self, project: Project, user_id: str):
        """Delete project folders from storage"""
        
        try:
            # Get project folder path using FileService logic
            project_folder = self.file_service.get_project_folder_path(
                user_id, 
                "user",  # We'll get the actual user name if needed
                str(project.id), 
                project.name
            )
            
            # Also try alternative path structure
            uploads_base = Path("uploads")
            user_folder = uploads_base / f"{user_id}_*"
            
            # Find and delete project folders
            folders_deleted = 0
            
            # Try direct project folder
            if project_folder.exists():
                shutil.rmtree(project_folder)
                folders_deleted += 1
                logger.info(f"Deleted project folder: {project_folder}")
            
            # Search for project folders in user directories
            for user_dir in uploads_base.glob(f"{user_id}_*"):
                if user_dir.is_dir():
                    project_dirs = user_dir.glob(f"**/projects/*{project.id}*")
                    for project_dir in project_dirs:
                        if project_dir.is_dir():
                            shutil.rmtree(project_dir)
                            folders_deleted += 1
                            logger.info(f"Deleted project folder: {project_dir}")
            
            logger.info(f"Deleted {folders_deleted} folders for project {project.id}")
            
        except Exception as e:
            logger.error(f"Error deleting project folders: {e}")
            # Don't raise - folder deletion failure shouldn't stop the process
    
    async def _delete_upload_sessions(self, db: AsyncSession, project_id: str):
        """Delete all upload sessions for the project"""
        
        try:
            stmt = delete(UploadSession).where(UploadSession.project_id == project_id)
            result = await db.execute(stmt)
            await db.commit()
            
            logger.info(f"Deleted {result.rowcount} upload sessions for project {project_id}")
            
        except Exception as e:
            logger.error(f"Error deleting upload sessions: {e}")
            raise
    
    async def _delete_project_collaborators(self, db: AsyncSession, project_id: str):
        """Delete all project collaborators"""
        
        try:
            stmt = delete(ProjectCollaborator).where(ProjectCollaborator.project_id == project_id)
            result = await db.execute(stmt)
            await db.commit()
            
            logger.info(f"Deleted {result.rowcount} collaborators for project {project_id}")
            
        except Exception as e:
            logger.error(f"Error deleting project collaborators: {e}")
            raise
    
    async def _delete_project_record(self, db: AsyncSession, project: Project):
        """Delete the project record itself"""
        
        try:
            await db.delete(project)
            await db.commit()
            
            logger.info(f"Deleted project record: {project.id}")
            
        except Exception as e:
            logger.error(f"Error deleting project record: {e}")
            raise
    
    def _update_deletion_status(
        self, 
        deletion_id: str, 
        status: str, 
        progress: float, 
        current_step: str
    ):
        """Update deletion status"""
        
        if deletion_id in self.active_deletions:
            self.active_deletions[deletion_id].update({
                "status": status,
                "progress": progress,
                "current_step": current_step,
                "updated_at": datetime.now()
            })
            
            if status == "completed":
                self.active_deletions[deletion_id]["completed_at"] = datetime.now()
            elif status == "failed":
                self.active_deletions[deletion_id]["failed_at"] = datetime.now()
    
    async def get_deletion_status(self, deletion_id: str) -> Optional[Dict[str, Any]]:
        """Get deletion status"""
        
        if deletion_id not in self.active_deletions:
            return None
        
        status = self.active_deletions[deletion_id].copy()
        
        # Add estimated time remaining
        if status["status"] == "in_progress":
            elapsed = (datetime.now() - status["started_at"]).total_seconds()
            if status["progress"] > 0:
                estimated_total = elapsed / (status["progress"] / 100)
                estimated_remaining = max(0, estimated_total - elapsed)
                status["estimated_remaining_seconds"] = int(estimated_remaining)
        
        return status
    
    def cleanup_completed_deletions(self):
        """Clean up completed deletion tracking (call periodically)"""
        
        current_time = datetime.now()
        to_remove = []
        
        for deletion_id, status in self.active_deletions.items():
            # Remove completed/failed deletions after 1 hour
            if status["status"] in ["completed", "failed"]:
                completion_time = status.get("completed_at") or status.get("failed_at")
                if completion_time and (current_time - completion_time).total_seconds() > 3600:
                    to_remove.append(deletion_id)
        
        for deletion_id in to_remove:
            del self.active_deletions[deletion_id]

# Global instance
background_project_deletion_service = BackgroundProjectDeletionService() 