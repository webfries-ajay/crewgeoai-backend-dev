#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path
from sqlalchemy import select

# Add current directory to path
sys.path.append('.')

from core.database import get_db_session
from models.processed_image import ProcessedImage

async def check_processed_images():
    try:
        async with get_db_session() as db:
            result = await db.execute(select(ProcessedImage))
            images = result.scalars().all()
            
            print(f'🔍 Found {len(images)} processed images in database:')
            print('=' * 80)
            
            missing_files = []
            total_ndvi = 0
            total_ndmi = 0
            
            for img in images:
                path = Path(img.processed_image_path) if img.processed_image_path else None
                exists = path.exists() if path else False
                
                if img.processed_image_type.value == 'ndvi':
                    total_ndvi += 1
                elif img.processed_image_type.value == 'ndmi':
                    total_ndmi += 1
                
                print(f'📁 Record: {img.id}')
                print(f'   📄 File ID: {img.file_id}')
                print(f'   🏷️  Type: {img.processed_image_type.value}')
                print(f'   📍 Path: {img.processed_image_path}')
                print(f'   ✅/❌ Exists: {"✅ YES" if exists else "❌ NO"}')
                print(f'   📝 Filename: {img.processed_filename}')
                print(f'   📊 Status: {img.processing_status}')
                print(f'   📅 Created: {img.created_at}')
                
                if not exists:
                    missing_files.append({
                        'id': img.id,
                        'file_id': img.file_id,
                        'type': img.processed_image_type.value,
                        'path': img.processed_image_path,
                        'filename': img.processed_filename
                    })
                
                print('   ' + '-' * 50)
            
            print('\n📊 SUMMARY:')
            print(f'   Total NDVI images: {total_ndvi}')
            print(f'   Total NDMI images: {total_ndmi}')
            print(f'   Missing files: {len(missing_files)}')
            
            if missing_files:
                print('\n🚨 MISSING FILES:')
                for missing in missing_files:
                    print(f'   ❌ {missing["type"].upper()} for file {missing["file_id"]}: {missing["path"]}')
            
            print('\n🔍 Checking thumbnail directories...')
            for img in images:
                if img.processed_image_path:
                    thumb_dir = Path(img.processed_image_path).parent / "thumbnails"
                    if thumb_dir.exists():
                        thumbs = list(thumb_dir.glob(f"{Path(img.processed_image_path).stem}_*.jpg"))
                        print(f'   📸 {img.processed_image_type.value} thumbs: {len(thumbs)} found')
                    else:
                        print(f'   📸 {img.processed_image_type.value} thumb dir missing: {thumb_dir}')
            
    except Exception as e:
        print(f'❌ Error checking processed images: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_processed_images()) 