# Large TIFF Upload Issue - Analysis and Solution

## Problem Identified

You reported that a 500MB TIFF file was uploaded but didn't show up in the processing logs. After analyzing the code, I found the root cause:

### Root Cause: Memory Exhaustion in Background Upload Service

The issue was in the `BackgroundUploadService` in `backend/services/background_upload_service.py`. The service was trying to read the entire 500MB TIFF file into memory at once:

```python
# PROBLEMATIC CODE (lines 95-97)
content = await file.read()  # This loads entire 500MB into RAM!
file_data.append({
    "filename": file.filename,
    "content": content,  # 500MB in memory
    "content_type": file.content_type,
    "size": len(content)
})
```

**What was happening:**
1. User uploads 500MB TIFF file
2. Background upload service tries to read entire file into memory
3. Memory allocation fails or times out
4. File processing never starts
5. No error logs because the failure happens before processing

## Solution Implemented

### 1. Enhanced Background Upload Service

**File:** `backend/services/background_upload_service.py`

**Key Changes:**
- **File size detection** before reading into memory
- **Streaming approach** for large files (>100MB)
- **Temporary file storage** for large files
- **Memory-efficient processing** pipeline

```python
# NEW CODE - Smart file handling
for file in files:
    try:
        # Check file size before reading
        file_size_mb = 0
        try:
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
```

### 2. New Streaming Method for Large Files

```python
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
```

### 3. Enhanced File Processing Pipeline

**File:** `backend/services/file_service.py`

**New Method:**
```python
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
```

### 4. Large File Processing Logic

The system now automatically detects large files and uses appropriate processing:

```python
# Check if this is a large file
is_large_file = file_info.get("is_large_file", False)

if is_large_file:
    # Handle large file from temporary storage
    return await self._process_large_file_from_temp(file_info, upload_session, db)
else:
    # Handle regular file from memory
    return await self._process_regular_file_from_data(file_info, upload_session, db)
```

## How It Works Now

### For Small Files (< 100MB):
1. Read entire file into memory
2. Process normally with existing pipeline
3. Create thumbnails and extract metadata

### For Large Files (> 100MB):
1. **Detect file size** before reading
2. **Stream to temporary storage** in chunks
3. **Move to final location** when processing
4. **Use chunked processing** for thumbnails
5. **Clean up temporary files** after processing

## Benefits

✅ **No more memory errors** with large TIFF files  
✅ **Automatic detection** of large files  
✅ **Streaming upload** prevents memory exhaustion  
✅ **Proper error handling** and cleanup  
✅ **Detailed logging** for debugging  
✅ **Backward compatibility** with existing files  

## Testing

I've created test scripts to verify the solution:

1. **`test_large_file_processing.py`** - Tests the large file processor
2. **`test_large_tiff_upload.py`** - Tests the background upload service

## Expected Behavior Now

When you upload a 500MB TIFF file, you should see logs like:

```
INFO: Processing file large_image.tiff: 500.0MB
INFO: Large file detected (500.0MB), using streaming approach: large_image.tiff
INFO: Successfully saved large file to temp: /tmp/large_upload_xxx/large_image.tiff
INFO: Processing large file from temp storage: large_image.tiff
INFO: Large file detected, using chunked processing: large_image.tiff
INFO: Image dimensions: 8000x6000, Estimated memory: 137.3MB
INFO: Creating thumbnail using chunked processing: /path/to/large_image.tiff
INFO: Very large image detected (137.3MB), using aggressive downsampling
INFO: ✅ Created thumbnail using aggressive downsampling: /path/to/thumbnails/large_image_large.jpg
INFO: Background processing completed for large file: large_image.tiff
INFO: Successfully uploaded large file: large_image.tiff
```

## Configuration

The system uses these thresholds (configurable):

- **Large file threshold**: 100MB
- **TIFF file threshold**: 50MB (more conservative)
- **Memory buffer**: 200MB (kept free for system)
- **Chunk size**: 8KB for streaming

## Next Steps

1. **Test the upload** with your 500MB TIFF file
2. **Monitor the logs** for the new processing messages
3. **Verify thumbnails** are created successfully
4. **Check file metadata** extraction works

The system should now handle your 500MB TIFF file without any memory issues and create thumbnails successfully! 