import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from models.file import File, FileType, FileStatus, UploadSession, UploadSessionStatus
from models.project import Project
from models.user import User
from services.file_service import FileService
from core.database import get_db

logger = logging.getLogger(__name__)

class BackgroundUploadService:
    """Service for handling background file uploads"""
    
    def __init__(self):
        self.file_service = FileService()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def create_upload_session(
        self,
        project_id: str,
        user_id: str,
        files: List[UploadFile],
        db: AsyncSession
    ) -> UploadSession:
        """Create a new upload session"""
        
        logger.info(f"Creating upload session for project {project_id}, user {user_id}")
        
        # Calculate total size and file count
        total_size = 0
        total_files = len(files)
        
        # Reset file positions to calculate size
        for file in files:
            file.file.seek(0, 2)  # Seek to end
            total_size += file.file.tell()
            file.file.seek(0)  # Reset to beginning
        
        # Create session name
        session_name = f"Upload {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create upload session
        upload_session = UploadSession(
            project_id=project_id,
            user_id=user_id,
            session_name=session_name,
            total_files=total_files,
            total_size=total_size,
            status=UploadSessionStatus.PENDING
        )
        
        db.add(upload_session)
        await db.commit()
        await db.refresh(upload_session)
        
        logger.info(f"Created upload session {upload_session.id} with {total_files} files ({total_size} bytes)")
        
        return upload_session
    
    async def start_background_upload(
        self,
        upload_session_id: str,
        files: List[UploadFile],
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Start background upload process"""
        
        logger.info(f"Starting background upload for session {upload_session_id}")
        
        # Store session info for tracking
        self.active_sessions[upload_session_id] = {
            "status": "starting",
            "progress": 0,
            "files_processed": 0,
            "started_at": datetime.now()
        }
        
        # Read file contents into memory before background processing
        # This is necessary because UploadFile objects get closed after the request
        file_data = []
        for file in files:
            try:
                # Check file size before reading
                file_size_mb = 0
                try:
                    # Try to get file size without reading content
                    file.file.seek(0, 2)  # Seek to end
                    file_size_mb = file.file.tell() / (1024 * 1024)
                    file.file.seek(0)  # Reset to beginning
                except:
                    pass
                
                logger.info(f"Processing file {file.filename}: {file_size_mb:.1f}MB")
                
                # For very large files (>100MB), use streaming approach
                if file_size_mb > 100:
                    logger.info(f"Large file detected ({file_size_mb:.1f}MB), using streaming approach: {file.filename}")
                    
                    # Save large file directly to disk first, then process
                    temp_file_path = await self._save_large_file_to_temp(file)
                    
                    file_data.append({
                        "filename": file.filename,
                        "content": None,  # Will be None for large files
                        "content_type": file.content_type,
                        "size": int(file_size_mb * 1024 * 1024),
                        "temp_file_path": str(temp_file_path),
                        "is_large_file": True
                    })
                    logger.info(f"Saved large file to temp: {temp_file_path}")
                else:
                    # For smaller files, read into memory as before
                    content = await file.read()
                    file_data.append({
                        "filename": file.filename,
                        "content": content,
                        "content_type": file.content_type,
                        "size": len(content),
                        "is_large_file": False
                    })
                    logger.info(f"Read file {file.filename}: {len(content)} bytes")
                    
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {e}")
                file_data.append({
                    "filename": file.filename,
                    "content": None,
                    "content_type": file.content_type,
                    "size": 0,
                    "error": str(e)
                })
        
        # Add background task with file data
        background_tasks.add_task(
            self._process_upload_session,
            upload_session_id,
            file_data
        )
        
        return {
            "success": True,
            "session_id": upload_session_id,
            "message": "Upload started in background",
            "total_files": len(files)
        }
    
    async def _process_upload_session(
        self,
        upload_session_id: str,
        file_data: List[Dict[str, Any]]
    ):
        """Process upload session in background"""
        
        logger.info(f"Processing upload session {upload_session_id}")
        
        async for db in get_db():
            try:
                # Get upload session
                session_stmt = select(UploadSession).options(
                    selectinload(UploadSession.project),
                    selectinload(UploadSession.user)
                ).where(UploadSession.id == upload_session_id)
                
                session_result = await db.execute(session_stmt)
                upload_session = session_result.scalar_one_or_none()
                
                if not upload_session:
                    logger.error(f"Upload session {upload_session_id} not found")
                    return
                
                # Update session status
                upload_session.status = UploadSessionStatus.UPLOADING
                upload_session.started_at = datetime.now()
                await db.commit()
                
                # Update tracking
                self.active_sessions[upload_session_id].update({
                    "status": "uploading",
                    "total_files": len(file_data)
                })
                
                # Process each file
                uploaded_files = []
                failed_files = []
                uploaded_size = 0
                
                for i, file_info in enumerate(file_data):
                    try:
                        filename = file_info["filename"]
                        logger.info(f"Processing file {i+1}/{len(file_data)}: {filename}")
                        
                        # Check if file has error from reading stage
                        if file_info.get("error"):
                            failed_files.append({
                                "filename": filename,
                                "error": file_info["error"]
                            })
                            continue
                        
                        # Process single file
                        result = await self._process_single_file_from_data(
                            file_info, upload_session, db
                        )
                        
                        if result["success"]:
                            uploaded_files.append(result["file"])
                            uploaded_size += result["file_size"]
                        else:
                            failed_files.append({
                                "filename": filename,
                                "error": result["error"]
                            })
                        
                        # Update progress
                        progress = ((i + 1) / len(file_data)) * 100
                        upload_session.upload_progress = progress
                        upload_session.uploaded_files = len(uploaded_files)
                        upload_session.failed_files = len(failed_files)
                        upload_session.uploaded_size = uploaded_size
                        
                        # Update tracking
                        self.active_sessions[upload_session_id].update({
                            "progress": progress,
                            "files_processed": i + 1,
                            "uploaded_files": len(uploaded_files),
                            "failed_files": len(failed_files)
                        })
                        
                        await db.commit()
                        
                        # Small delay to prevent overwhelming the system
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        filename = file_info.get("filename", "unknown")
                        logger.error(f"Error processing file {filename}: {e}")
                        failed_files.append({
                            "filename": filename,
                            "error": str(e)
                        })
                
                # Update final session status
                if failed_files and not uploaded_files:
                    upload_session.status = UploadSessionStatus.FAILED
                    upload_session.error_message = f"All {len(failed_files)} files failed to upload"
                elif failed_files:
                    upload_session.status = UploadSessionStatus.COMPLETED
                    upload_session.error_message = f"{len(failed_files)} files failed to upload"
                else:
                    upload_session.status = UploadSessionStatus.COMPLETED
                
                upload_session.completed_at = datetime.now()
                upload_session.failed_file_details = failed_files
                
                await db.commit()
                
                # Update project file statistics after successful upload
                if uploaded_files:  # Only update if files were successfully uploaded
                    try:
                        from services.project_service import ProjectService
                        project_service = ProjectService(db)
                        await project_service.update_project_file_statistics(upload_session.project_id)
                        logger.info(f"Updated project statistics after background upload: {len(uploaded_files)} files")
                    except Exception as e:
                        logger.error(f"Error updating project statistics after background upload: {str(e)}")
                        # Don't fail the upload if statistics update fails
                
                # Update tracking
                self.active_sessions[upload_session_id].update({
                    "status": "completed",
                    "progress": 100,
                    "completed_at": datetime.now(),
                    "uploaded_files": len(uploaded_files),
                    "failed_files": len(failed_files)
                })
                
                logger.info(f"Upload session {upload_session_id} completed: {len(uploaded_files)} uploaded, {len(failed_files)} failed")
                
            except Exception as e:
                logger.error(f"Error processing upload session {upload_session_id}: {e}")
                
                # Update session as failed
                try:
                    upload_session.status = UploadSessionStatus.FAILED
                    upload_session.error_message = str(e)
                    upload_session.completed_at = datetime.now()
                    await db.commit()
                except:
                    pass
                
                # Update tracking
                self.active_sessions[upload_session_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now()
                })
            
            finally:
                break
    
    async def _save_large_file_to_temp(self, file: UploadFile) -> Path:
        """Save large file to temporary storage using streaming"""
        import tempfile
        import aiofiles
        
        # Create temporary file
        temp_dir = Path(tempfile.mkdtemp(prefix="large_upload_"))
        temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        try:
            # Stream file content to disk
            async with aiofiles.open(temp_file, 'wb') as f:
                chunk_size = 8192  # 8KB chunks
                while chunk := await file.read(chunk_size):
                    await f.write(chunk)
            
            logger.info(f"Successfully saved large file to temp: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error saving large file to temp: {e}")
            # Cleanup on error
            if temp_file.exists():
                temp_file.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
            raise
    
    async def _process_single_file_from_data(
        self,
        file_info: Dict[str, Any],
        upload_session: UploadSession,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process a single file from file data"""
        
        filename = file_info["filename"]
        
        try:
            # Check if this is a large file
            is_large_file = file_info.get("is_large_file", False)
            
            if is_large_file:
                # Handle large file from temporary storage
                return await self._process_large_file_from_temp(file_info, upload_session, db)
            else:
                # Handle regular file from memory
                return await self._process_regular_file_from_data(file_info, upload_session, db)
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _process_large_file_from_temp(
        self,
        file_info: Dict[str, Any],
        upload_session: UploadSession,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process a large file from temporary storage"""
        
        filename = file_info["filename"]
        content_type = file_info["content_type"]
        file_size = file_info["size"]
        temp_file_path = Path(file_info["temp_file_path"])
        
        try:
            logger.info(f"Processing large file from temp storage: {filename}")
            
            # Determine file type from content type and filename
            file_type = self._determine_file_type(filename, content_type)
            
            # Create file record
            file_record = File(
                project_id=upload_session.project_id,
                user_id=upload_session.user_id,
                uploaded_by=upload_session.user_id,
                upload_session_id=upload_session.id,
                original_filename=filename,
                stored_filename=f"{uuid.uuid4()}_{filename}",
                file_path="",  # Will be set after saving
                file_type=file_type,
                mime_type=content_type,
                file_size=file_size,
                upload_status=FileStatus.UPLOADING
            )
            
            db.add(file_record)
            await db.flush()  # Get the ID
            
            # Move file from temp storage to final location
            file_path = await self.file_service.save_file_from_temp_storage(
                temp_file_path, filename, upload_session.project, upload_session.user, str(file_record.id)
            )
            
            # Update file record
            file_record.file_path = str(file_path)
            file_record.upload_status = FileStatus.UPLOADED
            
            await db.commit()
            
            # Schedule background processing for thumbnail generation and metadata extraction
            try:
                await self.file_service.process_file_background(str(file_record.id), str(file_path))
                logger.info(f"Background processing completed for large file: {filename}")
            except Exception as e:
                logger.error(f"Background processing failed for large file {filename}: {e}")
                # Don't fail the upload if processing fails
            
            # Cleanup temp file
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
                temp_dir = temp_file_path.parent
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp file {temp_file_path}: {cleanup_error}")
            
            logger.info(f"Successfully uploaded large file: {filename}")
            
            return {
                "success": True,
                "file": {
                    "id": str(file_record.id),
                    "filename": filename,
                    "size": file_size,
                    "type": file_type.value
                },
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Error processing large file {filename}: {e}")
            # Cleanup temp file on error
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
                temp_dir = temp_file_path.parent
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except:
                pass
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _process_regular_file_from_data(
        self,
        file_info: Dict[str, Any],
        upload_session: UploadSession,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process a regular file from memory data"""
        
        filename = file_info["filename"]
        
        try:
            # Validate file data
            file_content = file_info["content"]
            content_type = file_info["content_type"]
            file_size = file_info["size"]
            
            if file_content is None:
                raise Exception("File content is None")
            
            # Determine file type from content type and filename
            file_type = self._determine_file_type(filename, content_type)
            
            # Create file record
            file_record = File(
                project_id=upload_session.project_id,
                user_id=upload_session.user_id,
                uploaded_by=upload_session.user_id,  # Set uploaded_by to the same user
                upload_session_id=upload_session.id,
                original_filename=filename,
                stored_filename=f"{uuid.uuid4()}_{filename}",
                file_path="",  # Will be set after saving
                file_type=file_type,
                mime_type=content_type,
                file_size=file_size,
                upload_status=FileStatus.UPLOADING
            )
            
            db.add(file_record)
            await db.flush()  # Get the ID
            
            # Save file to storage from content
            file_path = await self.file_service.save_file_content_to_storage(
                file_content, filename, upload_session.project, upload_session.user, str(file_record.id)
            )
            
            # Update file record
            file_record.file_path = str(file_path)
            file_record.upload_status = FileStatus.UPLOADED
            
            await db.commit()
            
            # Schedule background processing for thumbnail generation and metadata extraction
            # Note: We can't use BackgroundTasks here since we're already in a background task
            # So we'll start the processing task directly
            try:
                await self.file_service.process_file_background(str(file_record.id), str(file_path))
                logger.info(f"Background processing completed for file: {filename}")
            except Exception as e:
                logger.error(f"Background processing failed for file {filename}: {e}")
                # Don't fail the upload if processing fails
            
            logger.info(f"Successfully uploaded file: {filename}")
            
            return {
                "success": True,
                "file": {
                    "id": str(file_record.id),
                    "filename": filename,
                    "size": file_size,
                    "type": file_type.value
                },
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _determine_file_type(self, filename: str, content_type: str) -> FileType:
        """Determine file type from filename and content type"""
        
        # Get file extension
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Image types
        if content_type and content_type.startswith('image/'):
            return FileType.IMAGE
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp']:
            return FileType.IMAGE
        
        # Video types
        if content_type and content_type.startswith('video/'):
            return FileType.VIDEO
        if ext in ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv']:
            return FileType.VIDEO
        
        # Document types
        if ext in ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt']:
            return FileType.DOCUMENT
        
        # Geospatial types
        if ext in ['shp', 'kml', 'kmz', 'gpx', 'geojson', 'gml', 'tiff', 'tif']:
            return FileType.GEOSPATIAL
        
        # Archive types
        if ext in ['zip', 'rar', '7z', 'tar', 'gz', 'bz2']:
            return FileType.ARCHIVE
        
        # Default to OTHER
        return FileType.OTHER
    
    async def _process_single_file(
        self,
        file: UploadFile,
        upload_session: UploadSession,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process a single file upload (legacy method for direct uploads)"""
        
        try:
            # Validate file
            file_info = await self.file_service.validate_file(file)
            
            # Create file record
            file_record = File(
                project_id=upload_session.project_id,
                user_id=upload_session.user_id,
                uploaded_by=upload_session.user_id,  # Set uploaded_by to the same user
                upload_session_id=upload_session.id,
                original_filename=file.filename,
                stored_filename=f"{uuid.uuid4()}_{file.filename}",
                file_path="",  # Will be set after saving
                file_type=FileType(file_info["type"]),
                mime_type=file_info["mime_type"],
                file_size=file_info["size"],
                upload_status=FileStatus.UPLOADING
            )
            
            db.add(file_record)
            await db.flush()  # Get the ID
            
            # Save file to storage
            file_path = await self.file_service.save_file_to_storage(
                file, upload_session.project, upload_session.user, str(file_record.id)
            )
            
            # Update file record
            file_record.file_path = str(file_path)
            file_record.upload_status = FileStatus.UPLOADED
            
            await db.commit()
            
            # Schedule background processing for thumbnail generation and metadata extraction
            try:
                await self.file_service.process_file_background(str(file_record.id), str(file_path))
                logger.info(f"Background processing completed for file: {file.filename}")
            except Exception as e:
                logger.error(f"Background processing failed for file {file.filename}: {e}")
                # Don't fail the upload if processing fails
            
            logger.info(f"Successfully uploaded file: {file.filename}")
            
            return {
                "success": True,
                "file": {
                    "id": str(file_record.id),
                    "filename": file.filename,
                    "size": file_info["size"],
                    "type": file_info["type"]
                },
                "file_size": file_info["size"]
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_session_status(
        self,
        session_id: str,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Get upload session status"""
        
        # Check in-memory tracking first
        if session_id in self.active_sessions:
            tracking_info = self.active_sessions[session_id]
        else:
            tracking_info = {}
        
        # Get from database
        session_stmt = select(UploadSession).where(UploadSession.id == session_id)
        session_result = await db.execute(session_stmt)
        upload_session = session_result.scalar_one_or_none()
        
        if not upload_session:
            return None
        
        return {
            "session_id": str(upload_session.id),
            "session_name": upload_session.session_name,
            "status": upload_session.status.value,
            "total_files": upload_session.total_files,
            "total_size": upload_session.total_size,
            "uploaded_files": upload_session.uploaded_files,
            "failed_files": upload_session.failed_files,
            "uploaded_size": upload_session.uploaded_size,
            "upload_progress": upload_session.upload_progress,
            "processing_progress": upload_session.processing_progress,
            "created_at": upload_session.created_at.isoformat() if upload_session.created_at else None,
            "started_at": upload_session.started_at.isoformat() if upload_session.started_at else None,
            "completed_at": upload_session.completed_at.isoformat() if upload_session.completed_at else None,
            "error_message": upload_session.error_message,
            "failed_file_details": upload_session.failed_file_details,
            # Include tracking info if available
            **tracking_info
        }
    
    async def get_user_active_sessions(
        self,
        user_id: str,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get all active upload sessions for a user"""
        
        sessions_stmt = select(UploadSession).where(
            UploadSession.user_id == user_id,
            UploadSession.status.in_([
                UploadSessionStatus.PENDING,
                UploadSessionStatus.UPLOADING,
                UploadSessionStatus.PROCESSING
            ])
        ).order_by(UploadSession.created_at.desc())
        
        sessions_result = await db.execute(sessions_stmt)
        sessions = sessions_result.scalars().all()
        
        result = []
        for session in sessions:
            status_info = await self.get_session_status(str(session.id), db)
            if status_info:
                result.append(status_info)
        
        return result

# Global instance
background_upload_service = BackgroundUploadService() 