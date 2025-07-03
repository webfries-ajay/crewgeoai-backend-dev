from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FastAPIFile, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import os
from pathlib import Path
import json
import asyncio
from sqlalchemy import select

from core.database import get_db
from core.security import get_current_user, get_optional_user, get_optional_user_with_query_token
from core.config import settings
from models.user import User
from models.file import File
from models.project import Project
from models.annotation import Annotation
from services.file_service import file_service
from services.background_upload_service import background_upload_service
from services.project_service import ProjectService
from schemas.file import FileResponse, FileListResponse, FileUploadResponse, UploadSessionResponse
from models.file_metadata import FileMetadata
# GeoAI imports moved to the endpoint to handle missing dependencies gracefully

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])

print("FILES ROUTER LOADED!")

# Global GeoAI agent instance for conversation continuity
geoai_agents = {}

@router.post("/upload/{project_id}", response_model=FileUploadResponse)
async def upload_files(
    project_id: str,
    request: Request,
    files: List[UploadFile] = FastAPIFile(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload multiple files to a project with background processing.
    
    - **project_id**: ID of the project to upload files to
    - **files**: List of files to upload (supports images, videos, documents, geospatial files)
    - **Returns**: Upload status with file information
    
    Features:
    - Multiple file upload support
    - Background processing for thumbnails and metadata extraction
    - Automatic file type detection
    - Virus scanning (placeholder)
    - Checksum calculation for integrity
    - Proper folder structure: uploads/{user_id}_{user_name}/projects/{project_id}_{project_name}/
    """
    # Log the incoming request details
    logger.info(f"File upload request received:")
    logger.info(f"  - Project ID: {project_id}")
    logger.info(f"  - User ID: {current_user.id}")
    logger.info(f"  - User Email: {current_user.email}")
    logger.info(f"  - Number of files: {len(files) if files else 0}")
    logger.info(f"  - Request headers: {dict(request.headers)}")
    
    # Log file details
    if files:
        for i, file in enumerate(files):
            logger.info(f"  - File {i+1}: {file.filename}, Size: {file.size if hasattr(file, 'size') else 'unknown'}, Type: {file.content_type}")
    
    try:
        if not files:
            logger.error("No files provided in request")
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 50:  # Limit to 50 files per upload
            logger.error(f"Too many files: {len(files)} (max 50)")
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per upload.")
        
        logger.info("Calling file_service.upload_files...")
        result = await file_service.upload_files(
            files=files,
            project_id=project_id,
            user_id=str(current_user.id),
            db=db,
            background_tasks=background_tasks
        )
        
        logger.info(f"File upload completed successfully: {result}")
        return result

    except HTTPException as e:
        logger.error(f"HTTP Exception in file upload: {e.status_code} - {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in file upload: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during file upload: {str(e)}"
        )

@router.post("/upload-background/{project_id}", response_model=UploadSessionResponse)
async def upload_files_background(
    project_id: str,
    request: Request,
    files: List[UploadFile] = FastAPIFile(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Start background file upload process.
    
    This endpoint creates an upload session and processes files in the background,
    allowing users to continue working while files are being uploaded.
    """
    
    logger.info(f"=== BACKGROUND UPLOAD REQUEST ===")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"User ID: {current_user.id}")
    logger.info(f"User Email: {current_user.email}")
    logger.info(f"Number of files: {len(files)}")
    
    # Log file details and calculate total size
    total_size = 0
    for i, file in enumerate(files):
        file.file.seek(0, 2)  # Seek to end to get size
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        total_size += size
        logger.info(f"File {i+1}: {file.filename} ({size} bytes, {file.content_type})")
    
    logger.info(f"Total upload size: {total_size} bytes")
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 100:  # Higher limit for background uploads
            raise HTTPException(status_code=400, detail="Too many files. Maximum 100 files per background upload.")
        
        # Create upload session
        upload_session = await background_upload_service.create_upload_session(
        project_id=project_id,
        user_id=str(current_user.id),
            files=files,
            db=db
        )
        
        # Start background upload
        result = await background_upload_service.start_background_upload(
            upload_session_id=str(upload_session.id),
            files=files,
            background_tasks=background_tasks
        )
        
        logger.info(f"Background upload started: {result}")
        
        return UploadSessionResponse(
            success=True,
            message="Upload started in background. You can continue working while files are being processed.",
            session_id=str(upload_session.id),
            session_name=upload_session.session_name,
            total_files=upload_session.total_files,
            total_size=upload_session.total_size,
            status="started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Background upload failed: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start background upload: {str(e)}")

@router.get("/upload-status/{session_id}")
async def get_upload_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get upload session status and progress"""
    
    try:
        status = await background_upload_service.get_session_status(session_id, db)
        
        if not status:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get upload status")

@router.get("/my-uploads")
async def get_my_active_uploads(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's active upload sessions"""
    
    try:
        sessions = await background_upload_service.get_user_active_sessions(
            str(current_user.id), db
        )
        
        return {
            "success": True,
            "active_sessions": sessions,
            "total_active": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error getting active uploads: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active uploads")

# IMPORTANT: Specific routes must come before generic ones to avoid route conflicts
@router.get("/view/{file_id}")
async def view_file(
    file_id: str,
    current_user: Optional[User] = Depends(get_optional_user_with_query_token),
    db: AsyncSession = Depends(get_db)
):
    """
    View/stream a file for display (images, videos, etc.).
    
    - **file_id**: ID of the file to view
    
    This endpoint streams the file content for viewing rather than downloading.
    Supports both Bearer token authentication and query parameter token authentication.
    """
    from fastapi.responses import StreamingResponse
    import aiofiles
    
    print("🔥 VIEW FILE ENDPOINT CALLED! 🔥")
    
    print(f"=== VIEW FILE REQUEST ===")
    print(f"File ID: {file_id}")
    print(f"Has current_user: {current_user is not None}")
    
    # If still no user, authentication failed
    if not current_user:
        print("ERROR: No authentication provided")
        raise HTTPException(status_code=401, detail="Authentication required")
    
    print(f"Authenticated user: {current_user.email} (ID: {current_user.id})")
    
    file_path = await file_service.get_file_path(
        file_id=file_id,
        user_id=str(current_user.id),
        db=db
    )
    
    if not file_path:
        print(f"ERROR: File not found or access denied: {file_id}")
        raise HTTPException(status_code=404, detail="File not found")
    
    print(f"File path resolved: {file_path}")
    
    # Get file info for proper content type
    try:
        file_details = await file_service.get_file_details(
        file_id=file_id,
        user_id=str(current_user.id),
        db=db
    )
    
        if not file_details:
            raise HTTPException(status_code=404, detail="File details not found")
        
        content_type = file_details.get('mime_type', 'application/octet-stream')
        filename = file_details.get('original_filename', 'file')
        
        print(f"Content type: {content_type}")
        print(f"Filename: {filename}")
        
    except Exception as e:
        print(f"ERROR: Error getting file details: {e}")
        content_type = 'application/octet-stream'
        filename = 'file'
    
    # Stream the file content
    async def file_streamer():
        try:
            # Get file size for adaptive chunk sizing
            file_size = file_path.stat().st_size
            
            # Adaptive chunk size: larger files get bigger chunks (8KB to 64KB)
            if file_size > 100 * 1024 * 1024:  # > 100MB
                chunk_size = 65536  # 64KB chunks for very large files
            elif file_size > 10 * 1024 * 1024:  # > 10MB
                chunk_size = 32768  # 32KB chunks for large files
            else:
                chunk_size = 8192   # 8KB chunks for smaller files
            
            print(f"Streaming {filename}: {file_size} bytes with {chunk_size} byte chunks")
            
            async with aiofiles.open(file_path, 'rb') as file:
                while chunk := await file.read(chunk_size):
                    yield chunk
        except Exception as e:
            print(f"ERROR: Error streaming file: {e}")
            raise HTTPException(status_code=500, detail="Error reading file")
    
    print(f"Starting file stream for: {filename}")
    
    # Get file size for Content-Length header
    file_size = file_path.stat().st_size
    
    return StreamingResponse(
        file_streamer(),
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename=\"{filename}\"",
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes"  # Enable range requests for large files
        }
    )

@router.get("/download/{file_id}")
async def download_file(
    file_id: str,
    token: Optional[str] = Query(None, description="Authentication token for direct access"),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download a file.
    
    - **file_id**: ID of the file to download
    - **token**: Optional authentication token for direct access (for img tags)
    
    Supports both Bearer token authentication and query parameter token authentication.
    """
    from fastapi.responses import FileResponse
    
    logger.info(f"=== DOWNLOAD FILE REQUEST ===")
    logger.info(f"File ID: {file_id}")
    logger.info(f"Has current_user: {current_user is not None}")
    logger.info(f"Has token: {token is not None}")
    
    # If no current_user from Bearer token, try token from query parameter
    if not current_user and token:
        try:
            logger.info("Attempting token authentication from query parameter")
            from core.security import security
            current_user = await security.get_current_user(db, token)
            logger.info(f"Token authentication successful: {current_user.email}")
        except Exception as e:
            logger.error(f"Token authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # If still no user, authentication failed
    if not current_user:
        logger.error("No authentication provided")
        raise HTTPException(status_code=401, detail="Authentication required")
    
    logger.info(f"Authenticated user: {current_user.email} (ID: {current_user.id})")
    
    file_path = await file_service.get_file_path(
        file_id=file_id,
        user_id=str(current_user.id),
        db=db
    )
    
    if not file_path:
        logger.error(f"File not found or access denied: {file_id}")
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.info(f"Download successful: {file_path}")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='application/octet-stream'
    )

@router.get("/thumbnail/{file_id}")
async def get_file_thumbnail(
    file_id: str,
    size: str = Query("medium", description="Thumbnail size: small, medium, large"),
    token: Optional[str] = Query(None, description="Authentication token for direct access"),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get thumbnail for an image or video file.
    
    - **file_id**: ID of the file
    - **size**: Thumbnail size (small, medium, large)
    - **token**: Optional authentication token for direct access (for img tags)
    
    Supports both Bearer token authentication and query parameter token authentication.
    """
    from fastapi.responses import FileResponse
    
    # If no current_user from Bearer token, try token from query parameter
    if not current_user and token:
        try:
            from core.security import security
            current_user = await security.get_current_user(db, token)
        except Exception as e:
            logger.error(f"Token authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # If still no user, authentication failed
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    thumbnail_path = await file_service.get_file_thumbnail(
        file_id=file_id,
        size=size,
        user_id=str(current_user.id),
        db=db
    )
    
    if not thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(path=thumbnail_path, media_type="image/jpeg")

@router.get("/file/{file_id}", response_model=FileResponse)
async def get_file_details(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific file"""
    
    result = await file_service.get_file_details(
        file_id=file_id,
        user_id=str(current_user.id),
        db=db
    )
    
    return result

@router.get("/processing-status/{project_id}")
async def get_processing_status(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get file processing status for a project"""
    
    result = await file_service.get_processing_status(
        project_id=project_id,
        user_id=str(current_user.id),
        db=db
    )
    
    return result

@router.post("/bulk-delete")
async def bulk_delete_files(
    file_ids: List[str],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Bulk delete multiple files"""
    
    if len(file_ids) > 100:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 100 files per bulk operation.")
    
    result = await file_service.bulk_delete_files(
        file_ids=file_ids,
        user_id=str(current_user.id),
        db=db
    )
    
    return result 

@router.delete("/file/{file_id}")
async def delete_file(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a single file"""
    
    result = await file_service.delete_file(
        file_id=file_id,
        user_id=str(current_user.id),
        db=db
    )
    
    return result

# IMPORTANT: This generic route must come LAST to avoid conflicts with specific routes above
@router.get("/{project_id}", response_model=FileListResponse)
async def get_project_files(
    project_id: str,
    skip: int = Query(0, ge=0, description="Number of files to skip"),
    limit: int = Query(100, ge=1, le=500, description="Number of files to return"),
    file_type: Optional[str] = Query(None, description="Filter by file type (image, video, document, geospatial)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all files in a project with pagination.
    
    - **project_id**: ID of the project
    - **skip**: Number of files to skip (for pagination)
    - **limit**: Maximum number of files to return (1-500)
    - **file_type**: Optional filter by file type
    """
    
    result = await file_service.get_project_files(
        project_id=project_id,
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
        file_type=file_type,
        db=db
    )
    
    return result

@router.post("/{file_id}/chat")
async def chat_with_geoai(
    file_id: str,
    request: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Chat with GeoAI about the current file using intelligent detection"""
    try:
        # Get project_id from request
        project_id = request.get("projectId")
        if not project_id:
            raise HTTPException(status_code=400, detail="projectId is required")

        # Fetch project and file in a single query for speed
        db_file = await db.get(File, file_id)
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")
        db_project = await db.get(Project, project_id)
        if not db_project or db_project.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        if str(db_file.project_id) != str(db_project.id):
            raise HTTPException(status_code=400, detail="File does not belong to the specified project")

        # Get category-specific master prompt
        from services.geoai.master_prompt_router import get_master_prompt
        # Handle both enum and string category values
        if hasattr(db_project.category, 'value'):
            category = db_project.category.value
        else:
            category = str(db_project.category) if db_project.category else "agriculture"
        master_prompt = get_master_prompt(category)
        
        # Extract request parameters first
        message = request.get("message", "").strip()
        text_only = request.get("textOnly", False)
        selected_file_ids = request.get("selectedFileIds", [])
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Debug prints for verification
        print(f"[GeoAI DEBUG] Project: {db_project.name}")
        print(f"[GeoAI DEBUG] Category: {category}")
        print(f"[GeoAI DEBUG] Primary File: {db_file.original_filename}")
        print(f"[GeoAI DEBUG] Selected Files: {len(selected_file_ids) if selected_file_ids else 0}")
        print(f"[GeoAI DEBUG] Text Only: {text_only}")
        print(f"[GeoAI DEBUG] Using category-specific master prompt")
        
        # Get file paths for image analysis (only if not text-only)
        file_paths = {}
        file_metadata = {}  # Store metadata for each file
        
        if not text_only:
            if selected_file_ids and len(selected_file_ids) > 0:
                # Multi-image analysis: get paths and metadata for all selected files
                print(f"[GeoAI DEBUG] Getting file paths and metadata for {len(selected_file_ids)} selected files")
                for file_id_item in selected_file_ids:
                    try:
                        file_path = await file_service.get_file_path(
                            file_id=file_id_item,
                            user_id=str(current_user.id),
                            db=db
                        )
                        if file_path:
                            # Get file info for the path
                            db_file_item = await db.get(File, file_id_item)
                            if db_file_item:
                                file_paths[file_id_item] = {
                                    'path': str(file_path),
                                    'filename': db_file_item.original_filename
                                }
                                
                                # Fetch metadata for this file
                                try:
                                    metadata_query = await db.execute(
                                        select(FileMetadata).where(FileMetadata.file_id == file_id_item)
                                    )
                                    metadata_result = metadata_query.scalar_one_or_none()
                                    
                                    if metadata_result:
                                        # Convert metadata to dict for easy access
                                        metadata_dict = {
                                            'camera_make': metadata_result.camera_make,
                                            'camera_model': metadata_result.camera_model,
                                            'focal_length': metadata_result.focal_length,
                                            'aperture': metadata_result.aperture,
                                            'iso_speed': metadata_result.iso_speed,
                                            'latitude': metadata_result.latitude,
                                            'longitude': metadata_result.longitude,
                                            'altitude': metadata_result.altitude,
                                            'date_taken': metadata_result.date_taken.isoformat() if metadata_result.date_taken else None,
                                            'image_quality': metadata_result.image_quality,
                                            'bit_depth': metadata_result.bit_depth,
                                            'is_professional_grade': metadata_result.is_professional_grade,
                                            'equipment_category': metadata_result.equipment_category,
                                            'has_gps': metadata_result.has_gps,
                                            'metadata_completeness_score': metadata_result.metadata_completeness_score,
                                            'extraction_confidence': metadata_result.extraction_confidence,
                                            'processed_metadata': metadata_result.processed_metadata
                                        }
                                        file_metadata[file_id_item] = metadata_dict
                                        print(f"[GeoAI DEBUG] Added metadata for: {db_file_item.original_filename}")
                                    else:
                                        print(f"[GeoAI DEBUG] No metadata found for: {db_file_item.original_filename}")
                                        file_metadata[file_id_item] = None
                                except Exception as metadata_error:
                                    print(f"[GeoAI DEBUG] Error fetching metadata for {file_id_item}: {str(metadata_error)}")
                                    file_metadata[file_id_item] = None
                                
                                print(f"[GeoAI DEBUG] Added file path: {db_file_item.original_filename}")
                    except Exception as e:
                        print(f"[GeoAI DEBUG] Warning: Could not get path for file {file_id_item}: {str(e)}")
            else:
                # Single image analysis: get path and metadata for primary file
                file_path = await file_service.get_file_path(
                    file_id=file_id,
                    user_id=str(current_user.id),
                    db=db
                )
                if file_path:
                    file_paths[file_id] = {
                        'path': str(file_path),
                        'filename': db_file.original_filename
                    }
                    
                    # Fetch metadata for primary file
                    try:
                        metadata_query = await db.execute(
                            select(FileMetadata).where(FileMetadata.file_id == file_id)
                        )
                        metadata_result = metadata_query.scalar_one_or_none()
                        
                        if metadata_result:
                            metadata_dict = {
                                'camera_make': metadata_result.camera_make,
                                'camera_model': metadata_result.camera_model,
                                'focal_length': metadata_result.focal_length,
                                'aperture': metadata_result.aperture,
                                'iso_speed': metadata_result.iso_speed,
                                'latitude': metadata_result.latitude,
                                'longitude': metadata_result.longitude,
                                'altitude': metadata_result.altitude,
                                'date_taken': metadata_result.date_taken.isoformat() if metadata_result.date_taken else None,
                                'image_quality': metadata_result.image_quality,
                                'bit_depth': metadata_result.bit_depth,
                                'is_professional_grade': metadata_result.is_professional_grade,
                                'equipment_category': metadata_result.equipment_category,
                                'has_gps': metadata_result.has_gps,
                                'metadata_completeness_score': metadata_result.metadata_completeness_score,
                                'extraction_confidence': metadata_result.extraction_confidence,
                                'processed_metadata': metadata_result.processed_metadata
                            }
                            file_metadata[file_id] = metadata_dict
                            print(f"[GeoAI DEBUG] Added metadata for primary file: {db_file.original_filename}")
                            print(f"[GeoAI DEBUG] Metadata content: {metadata_dict}")
                        else:
                            print(f"[GeoAI DEBUG] No metadata found for primary file: {db_file.original_filename}")
                            file_metadata[file_id] = None
                    except Exception as metadata_error:
                        print(f"[GeoAI DEBUG] Error fetching metadata for primary file: {str(metadata_error)}")
                        file_metadata[file_id] = None
        else:
            print(f"[GeoAI DEBUG] Text-only mode: skipping file path and metadata resolution")
        
        # For text-only conversations, we don't need file paths
        if not text_only and not file_paths:
            raise HTTPException(status_code=404, detail="No valid file paths found")
        
        # Use GeoAI agent with category-specific master prompt and conversation history
        from services.geoai.smart_geoai_agent import SmartGeoAIAgent
        agent = SmartGeoAIAgent(
            master_prompt=master_prompt,
            project_id=project_id,
            user_id=str(current_user.id)
        )
        
        # Set file paths and metadata for multi-image analysis
        agent.selected_file_paths = file_paths
        agent.selected_file_ids = selected_file_ids if selected_file_ids and not text_only else []
        agent.selected_file_metadata = file_metadata  # Add metadata to agent
        print(f"[GeoAI DEBUG] Agent configured with {len(file_paths)} file paths and metadata")
        
        # Set the current image path only if not text-only conversation
        if not text_only and file_paths:
            # Use the first file path as the primary image
            primary_file_id = list(file_paths.keys())[0]
            agent.current_image = file_paths[primary_file_id]['path']
            print(f"[GeoAI DEBUG] Primary image path set: {file_paths[primary_file_id]['filename']}")
        else:
            agent.current_image = None
            print(f"[GeoAI DEBUG] Text-only conversation mode")
        
        print(f"[GeoAI DEBUG] Master prompt loaded for category: {category}")
        print(f"[GeoAI DEBUG] Conversation history enabled for project: {project_id}")
        print(f"[GeoAI DEBUG] Starting streaming response")
        
        async def generate_stream():
            try:
                # Process the message with streaming (agent will intelligently detect what needs to be analyzed)
                async for chunk in agent.process_message(message):
                    # Format as SSE (Server-Sent Events)
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Send completion signal
                file_info = {
                    'filename': db_file.original_filename,
                    'file_type': db_file.file_type,
                    'size': f'{db_file.file_size / (1024*1024):.2f} MB' if db_file.file_size else 'Unknown',
                    'can_analyze': True,
                    'selected_files_count': len(selected_file_ids) if selected_file_ids else 1,
                    'text_only': text_only
                }
                yield f"data: {json.dumps({'done': True, 'file_info': file_info})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming GeoAI chat: {str(e)}")
                yield f"data: {json.dumps({'error': f'Chat error: {str(e)}'})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8",
            }
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GeoAI chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/{file_id}/chat/reset")
async def reset_geoai_chat(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reset the GeoAI conversation for this project"""
    try:
        # Get the file from database
        db_file = await db.get(File, file_id)
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user has access to this file's project
        if db_file.project_id:
            project_service = ProjectService(db)
            project = await project_service.get_project_by_id(str(db_file.project_id), current_user.id)
            if not project or project.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Reset the conversation using conversation manager
        project_id = str(project.id)
        user_id = str(current_user.id)
        
        from services.geoai.conversation_manager import conversation_manager
        result = conversation_manager.clear_conversation(project_id, user_id)
        
        # Also clean up old agent instances (backward compatibility)
        if project_id in geoai_agents:
            del geoai_agents[project_id]
        
        print(f"[GeoAI DEBUG] Conversation reset for project {project_id}, user {user_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting GeoAI chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")

@router.get("/{file_id}/chat/history")
async def get_chat_history(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the conversation history for this project"""
    try:
        # Get the file from database
        db_file = await db.get(File, file_id)
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user has access to this file's project
        if db_file.project_id:
            project_service = ProjectService(db)
            project = await project_service.get_project_by_id(str(db_file.project_id), current_user.id)
            if not project or project.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        project_id = str(project.id)
        user_id = str(current_user.id)
        
        from services.geoai.conversation_manager import conversation_manager
        
        # Get conversation history
        history = conversation_manager.get_conversation_history(project_id, user_id)
        
        # Get conversation stats
        stats = conversation_manager.get_conversation_stats(project_id, user_id)
        
        # Format history for frontend
        formatted_history = []
        for msg in history:
            if hasattr(msg, 'content'):
                msg_type = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                formatted_history.append({
                    "type": msg_type,
                    "content": msg.content,
                    "timestamp": None  # Add timestamp if available in future
                })
        
        return {
            "success": True,
            "history": formatted_history,
            "stats": stats,
            "project_id": project_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@router.get("/{file_id}/metadata", response_model=dict)
async def get_file_metadata(
    file_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed metadata for a specific file"""
    try:
        # Get the file
        file = await file_service.get_file_by_id(file_id, db)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user has access to this file's project
        if file.project_id:
            project_service = ProjectService(db)
            project = await project_service.get_project_by_id(str(file.project_id), current_user.id)
            if not project or project.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get file metadata
        metadata = await file_service.get_file_metadata(file_id, db)
        
        # Get file info
        file_info = {
            "id": str(file.id),
            "filename": file.original_filename,
            "file_type": file.file_type,
            "file_size": file.file_size,
            "mime_type": file.mime_type,
            "width": file.width,
            "height": file.height,
            "uploaded_at": file.created_at.isoformat() if file.created_at else None,
            "processing_status": file.processing_status,
            "upload_status": file.upload_status
        }
        
        return {
            "success": True,
            "file_info": file_info,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{file_id}/annotations")
async def get_file_annotations(
    file_id: str,
    annotation_type: Optional[str] = Query(None, description="Filter by annotation type"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get annotations for a specific file"""
    try:
        # Get the file
        file = await file_service.get_file_by_id(file_id, db)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user has access to this file's project
        if file.project_id:
            project_service = ProjectService(db)
            project = await project_service.get_project_by_id(str(file.project_id), current_user.id)
            if not project or project.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Query annotations
        stmt = select(Annotation).where(Annotation.file_id == file.id)
        
        if annotation_type:
            stmt = stmt.where(Annotation.annotation_type == annotation_type)
        
        result = await db.execute(stmt)
        annotations = result.scalars().all()
        
        # Format annotations for frontend
        formatted_annotations = []
        for annotation in annotations:
            formatted_annotations.append({
                "id": str(annotation.id),
                "annotation_type": annotation.annotation_type,
                "coordinates": annotation.coordinates,
                "properties": annotation.properties,
                "label": annotation.label,
                "created_by_ai": annotation.created_by_ai,
                "ai_agent": annotation.ai_agent,
                "confidence_score": annotation.confidence_score,
                "created_at": annotation.created_at.isoformat() if annotation.created_at else None
            })
        
        return {
            "success": True,
            "file_id": file_id,
            "annotations": formatted_annotations,
            "total_count": len(formatted_annotations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file annotations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 