import os
import uuid
import hashlib
import shutil
import asyncio
import aiofiles
import mimetypes
import gc
import psutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update, delete
from sqlalchemy.orm import selectinload
from fastapi import UploadFile, HTTPException, BackgroundTasks
from PIL import Image, ImageOps
import logging

# Configure PIL to handle very large images
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection for large images

from core.database import get_db
from core.config import settings
from models.file import File, FileType, FileStatus, FileThumbnail
from models.project import Project
from models.user import User
from services.metadata import ProductionMetadataExtractor
from models.file_metadata import FileMetadata

logger = logging.getLogger(__name__)

# Optional imports
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg-python not available. Video processing will be limited.")

class LargeFileProcessor:
    """Handles large file processing with memory management and chunking"""
    
    def __init__(self):
        self.max_memory_mb = 1024  # 1GB memory limit
        self.large_file_threshold_mb = 100  # Files over 100MB are considered large
        self.tiff_chunk_size = 2048  # 2K chunks for TIFF processing
        self.memory_buffer_mb = 200  # Keep 200MB buffer for system
        
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)
        except:
            return 1024  # Default to 1GB if psutil fails
    
    def estimate_image_memory_mb(self, width: int, height: int, mode: str = 'RGB') -> float:
        """Estimate memory usage for an image in MB"""
        bytes_per_pixel = {
            'RGB': 3,
            'RGBA': 4,
            'L': 1,
            'LA': 2,
            'CMYK': 4,
            'P': 1,
            '1': 1
        }
        bpp = bytes_per_pixel.get(mode, 3)
        return (width * height * bpp) / (1024 * 1024)
    
    def should_use_chunked_processing(self, file_path: Path) -> bool:
        """Determine if file should use chunked processing"""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Always use chunked processing for very large files
            if file_size_mb > self.large_file_threshold_mb:
                return True
            
            # Check available memory
            available_memory = self.get_available_memory_mb()
            if available_memory < self.memory_buffer_mb:
                return True
            
            # For TIFF files, be more conservative
            if file_path.suffix.lower() in ['.tiff', '.tif']:
                return file_size_mb > 50  # Lower threshold for TIFF
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking file size for chunked processing: {e}")
            return True  # Default to chunked processing on error
    
    def get_image_info_safe(self, file_path: Path) -> Dict[str, Any]:
        """Get image information without loading the entire image into memory"""
        try:
            with Image.open(file_path) as img:
                # Don't call img.load() - just get basic info
                info = {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'estimated_memory_mb': self.estimate_image_memory_mb(img.width, img.height, img.mode)
                }
                
                # Check for multi-page images
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    info['n_frames'] = img.n_frames
                    info['is_multipage'] = True
                else:
                    info['is_multipage'] = False
                
                return info
                
        except Exception as e:
            logger.error(f"Error getting image info for {file_path}: {e}")
            # For very large images, try a different approach
            if "decompression bomb" in str(e) or "exceeds limit" in str(e):
                logger.info(f"Very large image detected, using alternative info extraction: {file_path}")
                return self._get_image_info_alternative(file_path)
            raise
    
    def _get_image_info_alternative(self, file_path: Path) -> Dict[str, Any]:
        """Alternative method to get image info for very large images"""
        try:
            # Use a more conservative approach for very large images
            with Image.open(file_path) as img:
                # Get basic info without loading the image
                info = {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'estimated_memory_mb': self.estimate_image_memory_mb(img.width, img.height, img.mode),
                    'is_multipage': False
                }
                
                # Check for multi-page images
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    info['n_frames'] = img.n_frames
                    info['is_multipage'] = True
                
                return info
                
        except Exception as e:
            logger.error(f"Error in alternative image info extraction: {e}")
            # Return basic info based on file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return {
                'width': 0,  # Will be determined during processing
                'height': 0,
                'mode': 'RGB',
                'format': file_path.suffix.lower().replace('.', ''),
                'size_mb': file_size_mb,
                'estimated_memory_mb': file_size_mb * 2,  # Rough estimate
                'is_multipage': False
            }
    
    def create_thumbnail_chunked(self, file_path: Path, target_size: Tuple[int, int], 
                                output_path: Path, quality: int = 85) -> bool:
        """Create thumbnail using chunked processing for large files"""
        try:
            logger.info(f"Creating thumbnail using chunked processing: {file_path}")
            
            # Get image info first
            img_info = self.get_image_info_safe(file_path)
            
            # For very large images, use a more aggressive downsampling approach
            if img_info['estimated_memory_mb'] > 500:
                logger.info(f"Very large image detected ({img_info['estimated_memory_mb']:.1f}MB), using aggressive downsampling")
                return self._create_thumbnail_aggressive_downsampling(file_path, target_size, output_path, quality)
            
            # For moderately large images, use standard chunked approach
            return self._create_thumbnail_standard_chunked(file_path, target_size, output_path, quality)
            
        except Exception as e:
            logger.error(f"Error in chunked thumbnail creation: {e}")
            return False
    
    def _create_thumbnail_aggressive_downsampling(self, file_path: Path, target_size: Tuple[int, int], 
                                                output_path: Path, quality: int) -> bool:
        """Create thumbnail using aggressive downsampling for very large images"""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use PIL's thumbnail method with multiple passes for very large images
            with Image.open(file_path) as img:
                # Convert to RGB if needed (do this in chunks)
                if img.mode != 'RGB':
                    logger.info(f"Converting {img.mode} to RGB for thumbnail")
                    if img.mode in ('RGBA', 'LA'):
                        # Create white background
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'LA':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1])
                        img = rgb_img
                    else:
                        img = img.convert('RGB')
                
                # Calculate intermediate sizes for progressive downsampling
                current_size = img.size
                target_width, target_height = target_size
                
                # Progressive downsampling to avoid memory issues
                intermediate_sizes = []
                while current_size[0] > target_width * 2 or current_size[1] > target_height * 2:
                    new_width = max(target_width * 2, current_size[0] // 2)
                    new_height = max(target_height * 2, current_size[1] // 2)
                    intermediate_sizes.append((new_width, new_height))
                    current_size = (new_width, new_height)
                
                # Apply progressive downsampling
                for intermediate_size in intermediate_sizes:
                    logger.info(f"Downsampling to intermediate size: {intermediate_size}")
                    img.thumbnail(intermediate_size, Image.Resampling.LANCZOS)
                    gc.collect()  # Force garbage collection
                
                # Final resize to target size
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create final image with padding if needed
                final_img = Image.new('RGB', target_size, (255, 255, 255))
                paste_x = (target_width - img.width) // 2
                paste_y = (target_height - img.height) // 2
                final_img.paste(img, (paste_x, paste_y))
                
                # Save thumbnail
                logger.info(f"Attempting to save thumbnail to: {output_path}")
                logger.info(f"Output directory exists: {output_path.parent.exists()}")
                logger.info(f"Output directory: {output_path.parent}")
                
                # Convert path to string to avoid PIL path issues
                output_path_str = str(output_path)
                logger.info(f"Output path string: {output_path_str}")
                
                final_img.save(output_path_str, "JPEG", quality=quality, optimize=True)
                
                logger.info(f"✅ Created thumbnail using aggressive downsampling: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error in aggressive downsampling: {e}")
            # Try alternative approach for very large images
            if "decompression bomb" in str(e) or "exceeds limit" in str(e):
                logger.info(f"Trying alternative thumbnail creation for very large image: {file_path}")
                return self._create_thumbnail_very_large_image(file_path, target_size, output_path, quality)
            return False
    
    def _create_thumbnail_very_large_image(self, file_path: Path, target_size: Tuple[int, int], 
                                         output_path: Path, quality: int) -> bool:
        """Create thumbnail for extremely large images using sampling approach"""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating thumbnail for very large image using sampling: {file_path}")
            
            # For extremely large images, we'll create a simple placeholder thumbnail
            # This ensures the system doesn't crash and provides a basic preview
            target_width, target_height = target_size
            
            # Create a placeholder thumbnail with file info
            placeholder_img = Image.new('RGB', target_size, (240, 240, 240))
            
            # Add some basic information to the placeholder
            try:
                # Try to get basic file info
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                # Create a simple colored background based on file type
                if file_path.suffix.lower() in ['.tiff', '.tif']:
                    bg_color = (200, 220, 255)  # Light blue for TIFF
                else:
                    bg_color = (255, 240, 200)  # Light orange for other formats
                
                placeholder_img = Image.new('RGB', target_size, bg_color)
                
                # Add a simple border
                from PIL import ImageDraw
                draw = ImageDraw.Draw(placeholder_img)
                draw.rectangle([0, 0, target_width-1, target_height-1], outline=(100, 100, 100), width=2)
                
                # Add text if possible (for larger thumbnails)
                if target_width >= 200 and target_height >= 200:
                    try:
                        from PIL import ImageFont
                        # Try to use a default font
                        font_size = min(target_width // 20, 12)
                        try:
                            font = ImageFont.truetype("arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                        
                        # Add file type and size info
                        text_lines = [
                            f"Large {file_path.suffix.upper()[1:]}",
                            f"{file_size_mb:.0f}MB"
                        ]
                        
                        y_offset = target_height // 4
                        for line in text_lines:
                            # Get text size
                            bbox = draw.textbbox((0, 0), line, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            
                            # Center the text
                            x = (target_width - text_width) // 2
                            y = y_offset
                            
                            # Draw text with outline for better visibility
                            draw.text((x, y), line, fill=(50, 50, 50), font=font)
                            y_offset += text_height + 5
                            
                    except Exception as text_error:
                        logger.warning(f"Could not add text to placeholder: {text_error}")
                
            except Exception as info_error:
                logger.warning(f"Could not get file info for placeholder: {info_error}")
            
                            # Save the placeholder thumbnail
                output_path_str = str(output_path)
                placeholder_img.save(output_path_str, "JPEG", quality=quality, optimize=True)
            
            logger.info(f"✅ Created placeholder thumbnail for very large image: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating placeholder thumbnail: {e}")
            return False
    
    def _create_thumbnail_standard_chunked(self, file_path: Path, target_size: Tuple[int, int], 
                                         output_path: Path, quality: int) -> bool:
        """Create thumbnail using standard chunked approach"""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    if img.mode in ('RGBA', 'LA'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'LA':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1])
                        img = rgb_img
                    else:
                        img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create final image with padding
                final_img = Image.new('RGB', target_size, (255, 255, 255))
                paste_x = (target_size[0] - img.width) // 2
                paste_y = (target_size[1] - img.height) // 2
                final_img.paste(img, (paste_x, paste_y))
                
                # Save thumbnail
                output_path_str = str(output_path)
                final_img.save(output_path_str, "JPEG", quality=quality, optimize=True)
                
                logger.info(f"✅ Created thumbnail using standard chunked approach: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error in standard chunked approach: {e}")
            return False

class FileService:
    """Production-level file service with background processing and large file support"""
    
    def __init__(self):
        self.upload_base_path = Path(settings.upload_dir or "uploads")
        self.allowed_extensions = {
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg',
            # Videos
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp',
            # Documents
            '.pdf', '.doc', '.docx', '.txt', '.csv', '.xls', '.xlsx', '.ppt', '.pptx',
            # Geospatial
            '.geojson', '.kml', '.kmz', '.shp', '.shx', '.dbf', '.prj', '.gpx',
            # Archives
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
            # Other
            '.json', '.xml', '.yaml', '.yml'
        }
        self.max_file_size = 5 * 1024 * 1024 * 1024  # 5GB
        self.chunk_size = 8192  # 8KB chunks for streaming
        
        # Initialize metadata extractor and large file processor
        self.metadata_extractor = ProductionMetadataExtractor()
        self.large_file_processor = LargeFileProcessor()
        
        # Ensure upload directory exists
        self.upload_base_path.mkdir(parents=True, exist_ok=True)
    
    def detect_mime_type(self, filename: str, file_content: bytes = None) -> str:
        """Detect MIME type using mimetypes library and file extension"""
        # First try mimetypes library
        mime_type, _ = mimetypes.guess_type(filename)
        
        if mime_type:
            return mime_type
        
        # Fallback to extension-based detection
        extension = Path(filename).suffix.lower()
        
        # Common MIME types mapping
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.geojson': 'application/geo+json',
            '.kml': 'application/vnd.google-earth.kml+xml',
            '.kmz': 'application/vnd.google-earth.kmz',
            '.zip': 'application/zip',
            '.rar': 'application/x-rar-compressed',
            '.7z': 'application/x-7z-compressed',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip',
        }
        
        return mime_map.get(extension, 'application/octet-stream')
    
    def get_user_folder_path(self, user_id: str, user_full_name: str) -> Path:
        """Get user's upload folder path"""
        folder_name = f"{user_id}_{user_full_name}".replace(" ", "_").replace("/", "_")
        return self.upload_base_path / folder_name
    
    def get_project_folder_path(self, user_id: str, user_full_name: str, 
                               project_id: str, project_name: str) -> Path:
        """Get project's upload folder path"""
        user_folder = self.get_user_folder_path(user_id, user_full_name)
        project_folder_name = f"{project_id}_{project_name}".replace(" ", "_").replace("/", "_")
        return user_folder / "projects" / project_folder_name
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename while preserving extension"""
        file_ext = Path(original_filename).suffix
        unique_id = str(uuid.uuid4())
        return f"{unique_id}_{original_filename}".replace(" ", "_")
    
    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {self.max_file_size / (1024*1024*1024):.1f}GB"
            )
        
        # Detect MIME type using our custom method
        mime_type = self.detect_mime_type(file.filename)
        
        # Use content_type as fallback if available
        if not mime_type or mime_type == 'application/octet-stream':
            mime_type = file.content_type or "application/octet-stream"
        
        return {
            "filename": file.filename,
            "size": file_size,
            "mime_type": mime_type,
            "extension": file_ext
        }
    
    async def upload_files(
        self,
        files: List[UploadFile],
        project_id: str,
        user_id: str,
        db: AsyncSession,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Upload multiple files with background processing"""
        
        logger.info(f"FileService.upload_files called:")
        logger.info(f"  - Project ID: {project_id}")
        logger.info(f"  - User ID: {user_id}")
        logger.info(f"  - Number of files: {len(files)}")
        
        try:
            # Validate project and user
            logger.info("Validating project and user...")
            project_query = select(Project).options(selectinload(Project.owner)).where(
                and_(Project.id == project_id, Project.owner_id == user_id)
            )
            project_result = await db.execute(project_query)
            project = project_result.scalar_one_or_none()
            
            if not project:
                logger.error(f"Project not found: {project_id} for user {user_id}")
                raise HTTPException(status_code=404, detail="Project not found")
            
            user = project.owner
            if not user:
                logger.error(f"User not found for project: {project_id}")
                raise HTTPException(status_code=404, detail="User not found")
                
            logger.info(f"Project and user validated: {project.name} owned by {user.email}")
        except Exception as e:
            logger.error(f"Error during project/user validation: {type(e).__name__}: {str(e)}")
            raise
        
        # Create project folder
        logger.info("Creating project folder...")
        try:
            project_folder = self.get_project_folder_path(
                str(user.id), user.full_name, str(project.id), project.name
            )
            project_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Project folder created/verified: {project_folder}")
        except Exception as e:
            logger.error(f"Error creating project folder: {type(e).__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create project folder: {str(e)}")
        
        # Process each file
        uploaded_files = []
        failed_files = []
        
        logger.info(f"Processing {len(files)} files...")
        
        for i, file in enumerate(files):
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            try:
                # Validate file
                logger.info(f"Validating file: {file.filename}")
                file_info = await self.validate_file(file)
                logger.info(f"File validation successful: {file_info}")
                
                # Generate unique filename
                unique_filename = self.generate_unique_filename(file_info["filename"])
                file_path = project_folder / unique_filename
                
                # Create file record in database
                file_type = File.determine_file_type(file_info["filename"], file_info["mime_type"])
                
                db_file = File(
                    project_id=project.id,
                    user_id=user.id,
                    stored_filename=unique_filename,
                    original_filename=file_info["filename"],
                    file_path=str(file_path),
                    file_type=file_type,
                    mime_type=file_info["mime_type"],
                    file_size=file_info["size"],
                    uploaded_by=user.id,
                    upload_status=FileStatus.UPLOADING,
                    processing_status="pending"
                )
                
                db.add(db_file)
                await db.flush()  # Get the ID
                
                # Save file to disk
                await self._save_file_to_disk(file, file_path)
                
                # Update upload status
                db_file.upload_status = FileStatus.UPLOADED
                
                # Calculate checksums
                checksums = await self._calculate_checksums(file_path)
                db_file.checksum_md5 = checksums["md5"]
                db_file.checksum_sha256 = checksums["sha256"]
                
                uploaded_files.append({
                    "id": str(db_file.id),
                    "filename": db_file.original_filename,
                    "size": db_file.file_size,
                    "type": db_file.file_type,
                    "status": "uploaded"
                })
                
                # Schedule background processing
                background_tasks.add_task(
                    self.process_file_background,
                    str(db_file.id),
                    str(file_path)
                )
                
            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {str(e)}")
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        await db.commit()
        
        # Update project file statistics after successful upload
        try:
            from services.project_service import ProjectService
            project_service = ProjectService(db)
            await project_service.update_project_file_statistics(project.id)
            logger.info(f"Updated project statistics after uploading {len(uploaded_files)} files")
        except Exception as e:
            logger.error(f"Error updating project statistics: {str(e)}")
            # Don't fail the upload if statistics update fails
        
        return {
            "success": True,
            "message": f"Successfully uploaded {len(uploaded_files)} files. Processing in background.",
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "total_uploaded": len(uploaded_files),
            "total_failed": len(failed_files)
        }
    
    async def save_file_content_to_storage(
        self, 
        file_content: bytes, 
        filename: str, 
        project, 
        user, 
        file_id: str
    ) -> Path:
        """Save file content to storage and return the file path"""
        
        try:
            # Create project folder
            project_folder = self.get_project_folder_path(
                str(user.id), user.full_name, str(project.id), project.name
            )
            project_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(filename)
            file_path = project_folder / unique_filename
            
            # Save file content to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Saved file content to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file content {filename}: {e}")
            raise
    
    async def save_file_to_storage(self, file: UploadFile, project, user, file_id: str) -> Path:
        """Save uploaded file to storage and return the file path"""
        
        try:
            # Create project folder
            project_folder = self.get_project_folder_path(
                str(user.id), user.full_name, str(project.id), project.name
            )
            project_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(file.filename)
            file_path = project_folder / unique_filename
            
            # Save file to disk
            await self._save_file_to_disk(file, file_path)
            
            logger.info(f"Saved file to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            raise
    
    async def save_file_from_temp_storage(self, temp_file_path: Path, filename: str, project, user, file_id: str) -> Path:
        """Move file from temporary storage to final project location"""
        
        try:
            # Create project folder
            project_folder = self.get_project_folder_path(
                str(user.id), user.full_name, str(project.id), project.name
            )
            project_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(filename)
            file_path = project_folder / unique_filename
            
            # Move file from temp location to final location
            import shutil
            shutil.move(str(temp_file_path), str(file_path))
            
            logger.info(f"Moved file from temp storage to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error moving file from temp storage {filename}: {e}")
            raise
    
    async def _save_file_to_disk(self, file: UploadFile, file_path: Path) -> None:
        """Save uploaded file to disk with streaming"""
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(self.chunk_size):
                    await f.write(chunk)
        except Exception as e:
            # Cleanup partial file
            if file_path.exists():
                file_path.unlink()
            raise e
    
    async def _calculate_checksums(self, file_path: Path) -> Dict[str, str]:
        """Calculate MD5 and SHA256 checksums"""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(self.chunk_size):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return {
            "md5": md5_hash.hexdigest(),
            "sha256": sha256_hash.hexdigest()
        }
    
    async def process_file_background(self, file_id: str, file_path: str) -> None:
        """Background processing for uploaded files"""
        try:
            async for db in get_db():
                # Get file record
                file_query = select(File).where(File.id == file_id)
                result = await db.execute(file_query)
                db_file = result.scalar_one_or_none()
                
                if not db_file:
                    logger.error(f"File {file_id} not found for processing")
                    return
                
                # Update processing status
                db_file.processing_status = "processing"
                await db.commit()
                
                # Process based on file type
                if db_file.file_type == FileType.IMAGE.value:
                    await self._process_image(db_file, Path(file_path), db)
                elif db_file.file_type == FileType.VIDEO.value:
                    await self._process_video(db_file, Path(file_path), db)
                elif db_file.file_type == FileType.DOCUMENT.value:
                    await self._process_document(db_file, Path(file_path), db)
                elif db_file.file_type == FileType.GEOSPATIAL.value:
                    await self._process_geospatial(db_file, Path(file_path), db)
                
                # Update completion status
                db_file.processing_status = "processed"
                db_file.processed_at = datetime.utcnow()
                await db.commit()
                
                logger.info(f"Successfully processed file {file_id}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {str(e)}")
            # Update error status
            try:
                async for db in get_db():
                    await db.execute(
                        update(File)
                        .where(File.id == file_id)
                        .values(processing_status="failed")
                    )
                    await db.commit()
            except Exception:
                pass
    
    async def _process_image(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process image file - extract metadata, create thumbnails with large file support"""
        try:
            logger.info(f"Processing image {db_file.id}: {db_file.original_filename}")
            
            # Check if this is a large file that needs special handling
            if self.large_file_processor.should_use_chunked_processing(file_path):
                logger.info(f"Large file detected, using chunked processing: {db_file.original_filename}")
                await self._process_image_large(db_file, file_path, db)
            else:
                logger.info(f"Standard file processing: {db_file.original_filename}")
                await self._process_image_standard(db_file, file_path, db)
            
            # Extract comprehensive metadata using the metadata extractor
            logger.info(f"Extracting metadata for image {db_file.id}: {file_path}")
            try:
                # Create new async session for metadata extraction
                from core.database import get_db_session
                async with get_db_session() as metadata_db:
                    metadata_success = await self.metadata_extractor.extract_and_save_metadata(
                        str(file_path), str(db_file.id), metadata_db
                    )
                    
                    if metadata_success:
                        logger.info(f"✅ Successfully extracted metadata for {db_file.original_filename}")
                    else:
                        logger.warning(f"⚠️ Failed to extract metadata for {db_file.original_filename}")
            except Exception as e:
                logger.error(f"❌ Error extracting metadata for {db_file.original_filename}: {e}")
                # Continue processing even if metadata extraction fails
                
        except Exception as e:
            logger.error(f"Error processing image {db_file.id}: {str(e)}")
            raise e
    
    async def _process_image_standard(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process standard-sized images using traditional approach"""
        try:
            with Image.open(file_path) as img:
                # Extract basic image dimensions
                db_file.width = img.width
                db_file.height = img.height
                
                # Create thumbnails
                await self._create_thumbnails(db_file, img, file_path, db)
                
                db_file.thumbnail_generated = True
                db_file.preview_generated = True
                
        except Exception as e:
            logger.error(f"Error in standard image processing {db_file.id}: {str(e)}")
            raise e
    
    async def _process_image_large(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process large images using memory-efficient chunked approach"""
        try:
            logger.info(f"Processing large image with chunked approach: {db_file.original_filename}")
            
            # Get image info without loading the entire image
            img_info = self.large_file_processor.get_image_info_safe(file_path)
            
            # Set basic dimensions
            db_file.width = img_info['width']
            db_file.height = img_info['height']
            
            logger.info(f"Image dimensions: {img_info['width']}x{img_info['height']}, "
                       f"Estimated memory: {img_info['estimated_memory_mb']:.1f}MB")
            
            # Create thumbnails using chunked processing
            await self._create_thumbnails_chunked(db_file, file_path, db)
            
            db_file.thumbnail_generated = True
            db_file.preview_generated = True
            
            # Force garbage collection after processing large file
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in large image processing {db_file.id}: {str(e)}")
            raise e
    
    async def _process_video(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process video file - extract metadata, create thumbnails"""
        try:
            if not FFMPEG_AVAILABLE:
                logger.warning(f"ffmpeg not available. Skipping video processing for {db_file.id}")
                return
            
            # Get video metadata using ffmpeg
            probe = ffmpeg.probe(str(file_path))
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if video_stream:
                db_file.width = int(video_stream.get('width', 0))
                db_file.height = int(video_stream.get('height', 0))
                db_file.duration_seconds = float(probe['format'].get('duration', 0))
                db_file.frame_rate = eval(video_stream.get('r_frame_rate', '0/1'))
            
            # Create video thumbnail
            await self._create_video_thumbnail(db_file, file_path, db)
            
            db_file.thumbnail_generated = True
            db_file.preview_generated = True
            
        except Exception as e:
            logger.error(f"Error processing video {db_file.id}: {str(e)}")
            raise e
    
    async def _process_document(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process document file - extract metadata"""
        try:
            # Basic document processing - metadata extraction can be added here later
            logger.info(f"Processing document {db_file.id}: {db_file.original_filename}")
            
        except Exception as e:
            logger.error(f"Error processing document {db_file.id}: {str(e)}")
            raise e
    
    async def _process_geospatial(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process geospatial file - extract coordinates and metadata with large file support"""
        try:
            logger.info(f"Processing geospatial file {db_file.id}: {db_file.original_filename}")
            
            # Try to process as image first (many geospatial files are image-based)
            try:
                # Check if this is a large file that needs special handling
                if self.large_file_processor.should_use_chunked_processing(file_path):
                    logger.info(f"Large geospatial file detected, using chunked processing: {db_file.original_filename}")
                    await self._process_geospatial_large(db_file, file_path, db)
                else:
                    logger.info(f"Standard geospatial file processing: {db_file.original_filename}")
                    await self._process_geospatial_standard(db_file, file_path, db)
                    
            except Exception as img_error:
                logger.warning(f"Could not process geospatial file as image: {img_error}")
                # For non-image geospatial files, we can add specialized processing here later
                # (e.g., for .shp, .kml, .geojson files)
                
        except Exception as e:
            logger.error(f"Error processing geospatial file {db_file.id}: {str(e)}")
            raise e
    
    async def _process_geospatial_standard(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process standard-sized geospatial files using traditional approach"""
        try:
            with Image.open(file_path) as img:
                # Extract basic image dimensions
                db_file.width = img.width
                db_file.height = img.height
                
                # Create thumbnails for geospatial images
                await self._create_thumbnails(db_file, img, file_path, db)
                
                db_file.thumbnail_generated = True
                db_file.preview_generated = True
                
                # Extract metadata for geospatial images
                logger.info(f"Extracting metadata for geospatial image {db_file.id}: {file_path}")
                try:
                    # Create new async session for metadata extraction
                    from core.database import get_db_session
                    async with get_db_session() as metadata_db:
                        metadata_success = await self.metadata_extractor.extract_and_save_metadata(
                            str(file_path), str(db_file.id), metadata_db
                        )
                        
                        if metadata_success:
                            logger.info(f"✅ Successfully extracted metadata for geospatial file {db_file.original_filename}")
                        else:
                            logger.warning(f"⚠️ Failed to extract metadata for geospatial file {db_file.original_filename}")
                except Exception as e:
                    logger.error(f"❌ Error extracting metadata for geospatial file {db_file.original_filename}: {e}")
                    # Continue processing even if metadata extraction fails
                    
        except Exception as e:
            logger.error(f"Error in standard geospatial processing {db_file.id}: {str(e)}")
            raise e
    
    async def _process_geospatial_large(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Process large geospatial files using memory-efficient chunked approach"""
        try:
            logger.info(f"Processing large geospatial file with chunked approach: {db_file.original_filename}")
            
            # Get image info without loading the entire image
            img_info = self.large_file_processor.get_image_info_safe(file_path)
            
            # Set basic dimensions
            db_file.width = img_info['width']
            db_file.height = img_info['height']
            
            logger.info(f"Geospatial image dimensions: {img_info['width']}x{img_info['height']}, "
                       f"Estimated memory: {img_info['estimated_memory_mb']:.1f}MB")
            
            # Create thumbnails using chunked processing
            await self._create_thumbnails_chunked(db_file, file_path, db)
            
            db_file.thumbnail_generated = True
            db_file.preview_generated = True
            
            # Extract metadata for geospatial images
            logger.info(f"Extracting metadata for large geospatial image {db_file.id}: {file_path}")
            try:
                # Create new async session for metadata extraction
                from core.database import get_db_session
                async with get_db_session() as metadata_db:
                    metadata_success = await self.metadata_extractor.extract_and_save_metadata(
                        str(file_path), str(db_file.id), metadata_db
                    )
                    
                    if metadata_success:
                        logger.info(f"✅ Successfully extracted metadata for large geospatial file {db_file.original_filename}")
                    else:
                        logger.warning(f"⚠️ Failed to extract metadata for large geospatial file {db_file.original_filename}")
            except Exception as e:
                logger.error(f"❌ Error extracting metadata for large geospatial file {db_file.original_filename}: {e}")
                # Continue processing even if metadata extraction fails
            
            # Force garbage collection after processing large file
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in large geospatial processing {db_file.id}: {str(e)}")
            raise e
    
    async def _create_thumbnails(self, db_file: File, img: Image.Image, file_path: Path, db: AsyncSession) -> None:
        """Create thumbnails for images. This function NEVER modifies or overwrites the original image file.
        Preserves the entire image by scaling and padding, never crops."""
        thumbnail_sizes = [
            ("small", 150, 150),
            ("medium", 300, 300),
            ("large", 600, 600)
        ]
        
        thumbnails_dir = file_path.parent / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)
        
        for size_name, width, height in thumbnail_sizes:
            try:
                # Always work on a copy, never modify the original image
                thumb_img = img.copy()
                
                # Handle all image formats properly
                original_format = thumb_img.format
                logger.info(f"Processing {original_format or 'unknown'} format image")
                
                # Convert various formats to RGB for JPEG compatibility
                if thumb_img.mode in ('RGBA', 'LA', 'P', '1', 'L'):
                    if thumb_img.mode == 'P':
                        # Handle palette images
                        if 'transparency' in thumb_img.info:
                            thumb_img = thumb_img.convert('RGBA')
                        else:
                            thumb_img = thumb_img.convert('RGB')
                    
                    if thumb_img.mode in ('RGBA', 'LA'):
                        # Create white background for transparent images
                        rgb_img = Image.new('RGB', thumb_img.size, (255, 255, 255))
                        if thumb_img.mode == 'RGBA':
                            rgb_img.paste(thumb_img, mask=thumb_img.split()[-1])
                        elif thumb_img.mode == 'LA':
                            rgb_img.paste(thumb_img.convert('RGB'), mask=thumb_img.split()[-1])
                        thumb_img = rgb_img
                    elif thumb_img.mode in ('1', 'L'):
                        # Convert grayscale and 1-bit images to RGB
                        thumb_img = thumb_img.convert('RGB')
                elif thumb_img.mode == 'CMYK':
                    # Handle CMYK images (common in TIFF files)
                    thumb_img = thumb_img.convert('RGB')
                
                # Scale the image to fit within the target dimensions while preserving aspect ratio
                # This ensures the entire image is visible, no cropping
                thumb_img.thumbnail((width, height), Image.Resampling.LANCZOS)
                
                # Create a new image with exact target dimensions and white background
                final_img = Image.new('RGB', (width, height), (255, 255, 255))
                
                # Calculate position to center the scaled image
                paste_x = (width - thumb_img.width) // 2
                paste_y = (height - thumb_img.height) // 2
                
                # Paste the scaled image onto the centered position
                final_img.paste(thumb_img, (paste_x, paste_y))
                
                # Verify the final dimensions are exactly what we want
                if final_img.size != (width, height):
                    logger.error(f"Final image size mismatch: expected {(width, height)}, got {final_img.size}")
                    continue
                
                # Save thumbnail to a separate file, never overwrite the original
                thumb_filename = f"{db_file.original_filename}_{size_name}.jpg"
                thumb_path = thumbnails_dir / thumb_filename
                final_img.save(thumb_path, "JPEG", quality=85, optimize=True)
                
                # Verify the saved image dimensions
                with Image.open(thumb_path) as saved_img:
                    actual_width, actual_height = saved_img.size
                    if (actual_width, actual_height) != (width, height):
                        logger.error(f"Saved thumbnail dimensions incorrect: expected {(width, height)}, got {(actual_width, actual_height)}")
                        continue
                
                logger.info(f"✅ Created {size_name} thumbnail for {db_file.original_filename}: {thumb_path}")
                logger.info(f"📐 Target: {width}x{height}, Scaled image: {thumb_img.size}, Final: {actual_width}x{actual_height}")
                logger.info(f"🖼️  Original format: {original_format}, Preserved entire image with padding")
                logger.info(f"📂 Thumbnail directory: {thumbnails_dir}")
                
                thumbnail = FileThumbnail(
                    file_id=db_file.id,
                    thumbnail_type=size_name,
                    thumbnail_path=str(thumb_path),
                    width=actual_width,
                    height=actual_height,
                    file_size=thumb_path.stat().st_size
                )
                db.add(thumbnail)
                
            except Exception as e:
                logger.error(f"Error creating {size_name} thumbnail for {db_file.id}: {str(e)}")
                logger.error(f"Image mode: {img.mode}, Format: {img.format}, Size: {img.size}")
    
    async def _create_thumbnails_chunked(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Create thumbnails for large images using memory-efficient chunked processing"""
        thumbnail_sizes = [
            ("small", 150, 150),
            ("medium", 300, 300),
            ("large", 600, 600)
        ]
        
        thumbnails_dir = file_path.parent / "thumbnails"
        thumbnails_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating chunked thumbnails for large file: {db_file.original_filename}")
        
        for size_name, width, height in thumbnail_sizes:
            try:
                logger.info(f"Creating {size_name} thumbnail ({width}x{height}) for {db_file.original_filename}")
                
                # Generate thumbnail filename
                thumb_filename = f"{db_file.original_filename}_{size_name}.jpg"
                thumb_path = thumbnails_dir / thumb_filename
                
                # Use the large file processor to create thumbnail
                success = self.large_file_processor.create_thumbnail_chunked(
                    file_path, (width, height), thumb_path, quality=85
                )
                
                if success and thumb_path.exists():
                    # Verify the saved image dimensions
                    with Image.open(thumb_path) as saved_img:
                        actual_width, actual_height = saved_img.size
                        if (actual_width, actual_height) != (width, height):
                            logger.error(f"Saved thumbnail dimensions incorrect: expected {(width, height)}, got {(actual_width, actual_height)}")
                            continue
                    
                    # Create thumbnail record
                    thumbnail = FileThumbnail(
                        file_id=db_file.id,
                        thumbnail_type=size_name,
                        thumbnail_path=str(thumb_path),
                        width=actual_width,
                        height=actual_height,
                        file_size=thumb_path.stat().st_size
                    )
                    db.add(thumbnail)
                    
                    logger.info(f"✅ Created {size_name} thumbnail using chunked processing: {thumb_path}")
                    logger.info(f"📐 Target: {width}x{height}, Final: {actual_width}x{actual_height}")
                    
                else:
                    logger.error(f"Failed to create {size_name} thumbnail for {db_file.original_filename}")
                
                # Force garbage collection between thumbnails for large files
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error creating {size_name} thumbnail for {db_file.id}: {str(e)}")
                # Continue with other thumbnail sizes even if one fails
            
    
    async def _create_video_thumbnail(self, db_file: File, file_path: Path, db: AsyncSession) -> None:
        """Create thumbnail for video"""
        try:
            if not FFMPEG_AVAILABLE:
                logger.warning(f"ffmpeg not available. Skipping video thumbnail for {db_file.id}")
                return
            
            thumbnails_dir = file_path.parent / "thumbnails"
            thumbnails_dir.mkdir(exist_ok=True)
            
            thumb_filename = f"{db_file.original_filename}_thumb.jpg"
            thumb_path = thumbnails_dir / thumb_filename
            
            # Extract frame at 1 second
            (
                ffmpeg
                .input(str(file_path), ss=1)
                .output(str(thumb_path), vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            if thumb_path.exists():
                # Create thumbnail record
                thumbnail = FileThumbnail(
                    file_id=db_file.id,
                    thumbnail_type="video_thumb",
                    thumbnail_path=str(thumb_path),
                    width=300,
                    height=200,
                    file_size=thumb_path.stat().st_size
                )
                
                db.add(thumbnail)
                
        except Exception as e:
            logger.error(f"Error creating video thumbnail for {db_file.id}: {str(e)}")
    
    async def get_project_files(
        self,
        project_id: str,
        user_id: str,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        file_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get files for a project"""
        
        # Build query
        query = select(File).options(selectinload(File.thumbnails)).where(
            and_(File.project_id == project_id, File.uploaded_by == user_id)
        )
        
        if file_type:
            query = query.where(File.file_type == file_type)
        
        # Get total count
        count_query = select(func.count(File.id)).where(
            and_(File.project_id == project_id, File.uploaded_by == user_id)
        )
        if file_type:
            count_query = count_query.where(File.file_type == file_type)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Get files
        query = query.offset(skip).limit(limit).order_by(File.created_at.desc())
        # Eagerly load the metadata relationship to avoid lazy loading issues
        query = query.options(selectinload(File.file_metadata_record))
        result = await db.execute(query)
        files = result.scalars().all()
        
        return {
            "files": [file.to_dict(include_metadata=True) for file in files],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    
    async def delete_file(self, file_id: str, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Delete a file"""
        
        # Get file
        file_query = select(File).where(
            and_(File.id == file_id, File.uploaded_by == user_id)
        )
        result = await db.execute(file_query)
        db_file = result.scalar_one_or_none()
        
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete physical file
        file_path = Path(db_file.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete thumbnails
        thumbnails_dir = file_path.parent / "thumbnails"
        if thumbnails_dir.exists():
            for thumb_file in thumbnails_dir.glob(f"{db_file.original_filename}_*"):
                thumb_file.unlink()
        
        # Delete file metadata records
        await self._delete_file_metadata(db, file_id)
        
        # Store project ID before deleting file
        project_id = db_file.project_id
        
        # Delete from database
        await db.delete(db_file)
        await db.commit()
        
        # Update project file statistics after deletion
        try:
            from services.project_service import ProjectService
            project_service = ProjectService(db)
            await project_service.update_project_file_statistics(project_id)
            logger.info(f"Updated project statistics after deleting file {file_id}")
        except Exception as e:
            logger.error(f"Error updating project statistics after file deletion: {str(e)}")
            # Don't fail the deletion if statistics update fails
        
        return {
            "success": True,
            "message": "File deleted successfully"
        }
    
    async def get_file_details(self, file_id: str, user_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific file"""
        
        print(f"🔍 GET FILE DETAILS: file_id={file_id}, user_id={user_id}")
        
        # Use project-based authorization instead of uploader-based
        file_query = select(File).options(
            selectinload(File.thumbnails), 
            selectinload(File.project),
            selectinload(File.file_metadata_record)
        ).where(File.id == file_id)
        result = await db.execute(file_query)
        db_file = result.scalar_one_or_none()
        
        if not db_file:
            print(f"ERROR: File not found in get_file_details: {file_id}")
            return None
        
        if not db_file.project:
            print(f"ERROR: No project associated with file in get_file_details: {file_id}")
            return None
        
        print(f"File details found: {db_file.original_filename}")
        print(f"Project owner: {db_file.project.owner_id}, requesting user: {user_id}")
        
        # Check if user has access to the project (using UUID conversion like get_file_path)
        try:
            project_owner_uuid = uuid.UUID(str(db_file.project.owner_id))
            requesting_user_uuid = uuid.UUID(str(user_id))
            
            if project_owner_uuid != requesting_user_uuid:
                print(f"ERROR: Access denied in get_file_details: User {requesting_user_uuid} is not owner of project {db_file.project.id} (owner: {project_owner_uuid})")
                return None
                
        except ValueError as e:
            print(f"ERROR: Invalid UUID format in get_file_details: {e}")
            return None
        
        print(f"SUCCESS: File details authorization passed")
        
        file_dict = db_file.to_dict(include_metadata=True)
        print(f"File details: mime_type={file_dict.get('mime_type')}, filename={file_dict.get('original_filename')}")
        return file_dict
    
    async def get_file_path(self, file_id: str, user_id: str, db: AsyncSession) -> Optional[Path]:
        """Get file path for download"""
        
        print(f"=== FILE ACCESS DEBUG ===")
        print(f"Requested file_id: {file_id}")
        print(f"Requesting user_id: {user_id}")
        
        # Use project-based authorization instead of uploader-based
        file_query = select(File).options(selectinload(File.project)).where(File.id == file_id)
        result = await db.execute(file_query)
        db_file = result.scalar_one_or_none()
        
        if not db_file:
            print(f"ERROR: File not found: {file_id}")
            return None
        
        print(f"File found: {db_file.original_filename}")
        print(f"File project_id: {db_file.project_id}")
        print(f"File uploaded_by: {db_file.uploaded_by}")
        
        # Check if user has access to the project (either owner or collaborator)
        if not db_file.project:
            print(f"ERROR: No project associated with file {file_id}")
            return None
            
        print(f"Project found: {db_file.project.name} (ID: {db_file.project.id})")
        print(f"Project owner_id: {db_file.project.owner_id}")
        print(f"User requesting access: {user_id}")
        print(f"Owner check: {db_file.project.owner_id} == {user_id} = {db_file.project.owner_id == user_id}")
        
        # Convert both to UUID for proper comparison
        try:
            project_owner_uuid = uuid.UUID(str(db_file.project.owner_id))
            requesting_user_uuid = uuid.UUID(str(user_id))
            
            print(f"UUID comparison: {project_owner_uuid} == {requesting_user_uuid} = {project_owner_uuid == requesting_user_uuid}")
            
            if project_owner_uuid != requesting_user_uuid:
                print(f"ERROR: Access denied: User {requesting_user_uuid} is not owner of project {db_file.project.id} (owner: {project_owner_uuid})")
                return None
                
        except ValueError as e:
            print(f"ERROR: Invalid UUID format: {e}")
            return None
        
        print(f"Authorization passed for user {user_id}")
        
        file_path = Path(db_file.file_path)
        print(f"File path: {file_path}")
        print(f"File exists: {file_path.exists()}")
        
        if not file_path.exists():
            print(f"ERROR: File does not exist on disk: {file_path}")
            return None
        
        # Mark as accessed
        db_file.last_accessed_at = datetime.utcnow()
        await db.commit()
        
        print(f"SUCCESS: File access successful: {file_path}")
        return file_path
    
    async def get_file_thumbnail(self, file_id: str, size: str, user_id: str, db: AsyncSession) -> Optional[Path]:
        """Get thumbnail path for a file"""
        
        # Use project-based authorization instead of uploader-based
        file_query = select(File).options(selectinload(File.thumbnails), selectinload(File.project)).where(File.id == file_id)
        result = await db.execute(file_query)
        db_file = result.scalar_one_or_none()
        
        if not db_file:
            return None
        
        if not db_file.project:
            return None
        
        # Check if user has access to the project (using UUID conversion like get_file_path)
        try:
            project_owner_uuid = uuid.UUID(str(db_file.project.owner_id))
            requesting_user_uuid = uuid.UUID(str(user_id))
            
            if project_owner_uuid != requesting_user_uuid:
                return None
                
        except ValueError:
            return None
        
        # Find thumbnail of requested size
        for thumbnail in db_file.thumbnails:
            if thumbnail.thumbnail_type == size or (size == "video_thumb" and thumbnail.thumbnail_type == "video_thumb"):
                thumb_path = Path(thumbnail.thumbnail_path)
                if thumb_path.exists():
                    return thumb_path
        
        return None
    
    async def get_processing_status(self, project_id: str, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Get processing status for all files in a project"""
        
        # Get all files for the project
        files_query = select(File).where(
            and_(File.project_id == project_id, File.uploaded_by == user_id)
        )
        result = await db.execute(files_query)
        files = result.scalars().all()
        
        # Count files by status
        total_files = len(files)
        completed_files = sum(1 for f in files if f.processing_status == FileStatus.COMPLETED.value)
        processing_files = sum(1 for f in files if f.processing_status == FileStatus.PROCESSING.value)
        failed_files = sum(1 for f in files if f.processing_status == FileStatus.FAILED.value)
        pending_files = sum(1 for f in files if f.processing_status == FileStatus.PENDING.value)
        
        # Create file status list
        file_statuses = []
        for file in files:
            progress = None
            if file.processing_status == FileStatus.COMPLETED.value:
                progress = 100
            elif file.processing_status == FileStatus.PROCESSING.value:
                progress = 50  # Rough estimate
            elif file.processing_status == FileStatus.PENDING.value:
                progress = 0
            
            file_statuses.append({
                "file_id": file.id,
                "filename": file.original_filename,
                "upload_status": file.upload_status,
                "processing_status": file.processing_status,
                "progress_percentage": progress,
                "error_message": None  # Error messages will be available through metadata relationship if needed
            })
        
        return {
            "project_id": project_id,
            "total_files": total_files,
            "completed_files": completed_files,
            "processing_files": processing_files,
            "failed_files": failed_files,
            "pending_files": pending_files,
            "files": file_statuses
        }
    
    async def bulk_delete_files(self, file_ids: List[str], user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Delete multiple files at once"""
        
        deleted_count = 0
        failed_count = 0
        errors = []
        project_ids_to_update = set()  # Track projects that need statistics update
        
        for file_id in file_ids:
            try:
                # Get file
                file_query = select(File).where(
                    and_(File.id == file_id, File.uploaded_by == user_id)
                )
                result = await db.execute(file_query)
                db_file = result.scalar_one_or_none()
                
                if not db_file:
                    failed_count += 1
                    errors.append(f"File {file_id} not found")
                    continue
                
                # Delete physical file
                file_path = Path(db_file.file_path)
                if file_path.exists():
                    file_path.unlink()
                
                # Delete thumbnails
                thumbnails_dir = file_path.parent / "thumbnails"
                if thumbnails_dir.exists():
                    for thumb_file in thumbnails_dir.glob(f"{db_file.original_filename}_*"):
                        thumb_file.unlink()
                
                # Delete file metadata records
                await self._delete_file_metadata(db, file_id)
                
                # Store project ID for statistics update
                project_ids_to_update.add(db_file.project_id)
                
                # Delete from database
                await db.delete(db_file)
                deleted_count += 1
                
            except Exception as e:
                failed_count += 1
                errors.append(f"Error deleting file {file_id}: {str(e)}")
                logger.error(f"Error deleting file {file_id}: {str(e)}")
        
        await db.commit()
        
        # Update project file statistics for all affected projects
        for project_id in project_ids_to_update:
            try:
                from services.project_service import ProjectService
                project_service = ProjectService(db)
                await project_service.update_project_file_statistics(project_id)
                logger.info(f"Updated project statistics after bulk deletion for project {project_id}")
            except Exception as e:
                logger.error(f"Error updating project statistics after bulk deletion: {str(e)}")
                # Don't fail the deletion if statistics update fails
        
        return {
            "success": failed_count == 0,
            "message": f"Deleted {deleted_count} files, {failed_count} failed",
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "errors": errors
        }

    async def _delete_file_metadata(self, db: AsyncSession, file_id: str):
        """Delete all metadata records for a file"""
        try:
            stmt = delete(FileMetadata).where(FileMetadata.file_id == file_id)
            result = await db.execute(stmt)
            
            if result.rowcount > 0:
                logger.debug(f"Deleted {result.rowcount} metadata records for file {file_id}")
            
        except Exception as e:
            logger.error(f"Error deleting metadata for file {file_id}: {e}")
            # Don't fail the file deletion if metadata deletion fails

    async def get_file_metadata(self, file_id: str, db: AsyncSession) -> dict:
        """Get metadata for a specific file"""
        try:
            # Get the file metadata record
            stmt = select(FileMetadata).where(FileMetadata.file_id == file_id)
            result = await db.execute(stmt)
            metadata_record = result.scalar_one_or_none()
            
            if not metadata_record:
                return {
                    "message": "No metadata found for this file",
                    "has_metadata": False
                }
            
            # Convert to dictionary and clean up None values
            metadata_dict = {
                "has_metadata": True,
                "camera_info": {
                    "make": metadata_record.camera_make,
                    "model": metadata_record.camera_model,
                    "lens_model": metadata_record.lens_model
                },
                "technical_settings": {
                    "focal_length": metadata_record.focal_length,
                    "aperture": metadata_record.aperture,
                    "shutter_speed": metadata_record.shutter_speed,
                    "iso": metadata_record.iso_speed,
                    "white_balance": metadata_record.white_balance,
                    "flash_used": metadata_record.flash_used,
                    "exposure_mode": metadata_record.exposure_mode,
                    "metering_mode": metadata_record.metering_mode
                },
                "gps_data": {
                    "latitude": metadata_record.latitude,
                    "longitude": metadata_record.longitude,
                    "altitude": metadata_record.altitude,
                    "gps_precision": metadata_record.gps_precision,
                    "has_gps": metadata_record.has_gps
                },
                "image_details": {
                    "color_space": metadata_record.color_space,
                    "image_quality": metadata_record.image_quality,
                    "compression_ratio": metadata_record.compression_ratio,
                    "bit_depth": metadata_record.bit_depth
                },
                "date_time": {
                    "date_taken": metadata_record.date_taken.isoformat() if metadata_record.date_taken else None,
                    "date_modified": metadata_record.date_modified.isoformat() if metadata_record.date_modified else None,
                    "timezone_offset": metadata_record.timezone_offset
                },
                "equipment_info": {
                    "is_professional_grade": metadata_record.is_professional_grade,
                    "equipment_category": metadata_record.equipment_category
                },
                "content_analysis": {
                    "has_faces": metadata_record.has_faces,
                    "has_text": metadata_record.has_text,
                    "has_vehicles": metadata_record.has_vehicles,
                    "has_buildings": metadata_record.has_buildings,
                    "has_vegetation": metadata_record.has_vegetation
                },
                "quality_metrics": {
                    "metadata_completeness_score": metadata_record.metadata_completeness_score,
                    "extraction_confidence": metadata_record.extraction_confidence
                },
                "processing_info": {
                    "extraction_method": metadata_record.extraction_method,
                    "processing_time_ms": metadata_record.processing_time_ms,
                    "extraction_errors": metadata_record.extraction_errors,
                    "llm_ready": metadata_record.llm_ready,
                    "token_count_estimate": metadata_record.token_count_estimate
                },
                "raw_data": {
                    "raw_exif_data": metadata_record.raw_exif_data,
                    "raw_iptc_data": metadata_record.raw_iptc_data,
                    "raw_xmp_data": metadata_record.raw_xmp_data,
                    "processed_metadata": metadata_record.processed_metadata
                },
                "metadata_summary": metadata_record.metadata_summary,
                "extraction_timestamp": metadata_record.created_at.isoformat() if metadata_record.created_at else None
            }
            
            # Remove None values for cleaner output
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items() if v is not None}
                elif isinstance(d, list):
                    return [clean_dict(item) for item in d if item is not None]
                else:
                    return d
            
            return clean_dict(metadata_dict)
            
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return {
                "message": f"Error retrieving metadata: {str(e)}",
                "has_metadata": False
            }

    async def get_file_by_id(self, file_id: str, db: AsyncSession) -> Optional[File]:
        """Get a file by ID without user authorization (authorization handled in router)"""
        try:
            file_query = select(File).where(File.id == file_id)
            result = await db.execute(file_query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting file by ID {file_id}: {e}")
            return None

# Create service instance
file_service = FileService() 