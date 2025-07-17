import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.mask import mask
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from skimage import segmentation, measure
from sqlalchemy.ext.asyncio import AsyncSession
from models.file import File
from models.processed_image import ProcessedImage, ProcessedImageType
from sqlalchemy import select, func
import asyncio
import concurrent.futures
import time
from functools import partial
import gc
import psutil
import math

logger = logging.getLogger(__name__)

def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return 0.0

# === CHUNKED PROCESSING CONFIGURATION ===
CHUNK_SIZE_MB = 50  # Process in 50MB chunks
MAX_MEMORY_MB = 200  # Maximum memory usage
MIN_CHUNK_PIXELS = 1024 * 1024  # Minimum 1M pixels per chunk
MAX_CHUNK_PIXELS = 4 * 1024 * 1024  # Maximum 4M pixels per chunk

THUMBNAIL_SIZES = [
    ("small", 150, 150),
    ("medium", 300, 300),
    ("large", 600, 600),
]

# Production-level color maps for NDVI/NDMI visualization
NDVI_COLORMAP = [
    (0.0, [139, 69, 19]),    # Brown - Very poor vegetation
    (0.2, [255, 165, 0]),    # Orange - Poor vegetation  
    (0.3, [255, 255, 0]),    # Yellow - Moderate vegetation
    (0.5, [173, 255, 47]),   # Yellow-green - Good vegetation
    (0.7, [0, 255, 0]),      # Green - Very good vegetation
    (0.9, [0, 100, 0]),      # Dark green - Excellent vegetation
    (1.0, [0, 50, 0])        # Very dark green - Maximum vegetation
]

NDMI_COLORMAP = [
    (0.0, [139, 69, 19]),    # Brown - Very dry
    (0.2, [255, 140, 0]),    # Dark orange - Dry
    (0.3, [255, 215, 0]),    # Gold - Moderate moisture
    (0.5, [255, 255, 0]),    # Yellow - Good moisture
    (0.7, [173, 216, 230]),  # Light blue - High moisture
    (0.9, [0, 191, 255]),    # Deep sky blue - Very high moisture
    (1.0, [0, 0, 139])       # Navy blue - Maximum moisture
]

def _create_production_colormap(colormap_data: List[Tuple[float, List[int]]], size: int = 256) -> np.ndarray:
    """Create a production-level colormap for NDVI/NDMI visualization (optimized)"""
    print(f"⚡ [FAST COLORMAP] Creating optimized colormap with {size} colors...")
    
    colormap = np.zeros((size, 3), dtype=np.uint8)
    
    # Vectorized approach for better performance
    indices = np.arange(size)
    normalized_vals = indices / (size - 1)
    
    # Extract values and colors for vectorized interpolation
    values = np.array([item[0] for item in colormap_data])
    colors = np.array([item[1] for item in colormap_data])
    
    for i, normalized_val in enumerate(normalized_vals):
        # Find the appropriate color segment using vectorized operations
        valid_indices = np.where(values <= normalized_val)[0]
        if len(valid_indices) == 0:
            # Use first color
            colormap[i] = colors[0]
        elif len(valid_indices) == len(values):
            # Use last color
            colormap[i] = colors[-1]
        else:
            # Interpolate between appropriate colors
            j = valid_indices[-1]
            if j + 1 < len(values):
                val1, val2 = values[j], values[j + 1]
                color1, color2 = colors[j], colors[j + 1]
                
                # Linear interpolation
                t = (normalized_val - val1) / (val2 - val1) if val2 != val1 else 0
                colormap[i] = color1 + t * (color2 - color1)
            else:
                colormap[i] = colors[j]
    
    return colormap

def _apply_color_mapping(index_array: np.ndarray, index_type: str) -> np.ndarray:
    """Apply production-level color mapping to NDVI/NDMI arrays"""
    print(f"[COLOR MAPPING] Applying {index_type} color mapping...")
    
    # Choose colormap based on index type
    colormap_data = NDVI_COLORMAP if index_type == "ndvi" else NDMI_COLORMAP
    colormap = _create_production_colormap(colormap_data)
    
    # Normalize index from [-1, 1] to [0, 255]
    normalized = np.clip((index_array + 1) * 127.5, 0, 255).astype(np.uint8)
    
    # Apply colormap
    colored_array = np.zeros((*index_array.shape, 3), dtype=np.uint8)
    colored_array = colormap[normalized]
    
    print(f"[COLOR MAPPING] Color mapping applied. Output shape: {colored_array.shape}")
    return colored_array

def _calculate_comprehensive_statistics(index_array: np.ndarray, index_type: str) -> Dict[str, Any]:
    """Calculate comprehensive statistics for NDVI/NDMI with optimized performance"""
    print(f"⚡ [FAST STATS] Calculating optimized statistics for {index_type}...")
    
    # Remove NaN values for calculations (optimized with boolean indexing)
    valid_mask = ~np.isnan(index_array)
    valid_data = index_array[valid_mask]
    
    if len(valid_data) == 0:
        return {"error": "No valid data for statistics calculation"}
    
    # Optimized basic statistics - calculate all at once where possible
    print(f"⚡ [FAST STATS] Computing basic statistics...")
    
    # Use vectorized operations for speed
    basic_stats = {
        "count": len(valid_data),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)), 
        "std": float(np.std(valid_data)),
        "variance": float(np.var(valid_data)),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "range": float(np.ptp(valid_data))  # Peak-to-peak (optimized range)
    }
    
    # Optimized percentiles calculation (single call for all percentiles)
    print(f"⚡ [FAST STATS] Computing percentiles...")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = np.percentile(valid_data, percentiles)
    percentile_stats = {f"p{p}": float(val) for p, val in zip(percentiles, percentile_values)}
    
    # Index-specific classifications and thresholds
    if index_type == "ndvi":
        # NDVI thresholds for vegetation classification
        thresholds = {
            "bare_soil": (-1.0, 0.1),
            "sparse_vegetation": (0.1, 0.3),
            "moderate_vegetation": (0.3, 0.5),
            "dense_vegetation": (0.5, 0.7),
            "very_dense_vegetation": (0.7, 1.0)
        }
        
        # Optimized classification calculation using vectorized operations
        print(f"⚡ [FAST STATS] Computing NDVI classifications...")
        classifications = {}
        total_count = len(valid_data)
        
        for class_name, (min_val, max_val) in thresholds.items():
            # Use vectorized operations for speed
            mask = (valid_data >= min_val) & (valid_data < max_val)
            count = np.count_nonzero(mask)  # Faster than np.sum for boolean
            percentage = (count / total_count) * 100
            classifications[class_name] = {
                "percentage": round(percentage, 2),
                "pixel_count": int(count),
                "threshold": (min_val, max_val)
            }
        
        # Optimized health indicators calculation
        print(f"⚡ [FAST STATS] Computing NDVI health indicators...")
        healthy_vegetation = np.count_nonzero(valid_data >= 0.3)
        stressed_vegetation = np.count_nonzero((valid_data >= 0.1) & (valid_data < 0.3))
        health_score = (healthy_vegetation / total_count) * 100
        
        health_indicators = {
            "healthy_vegetation_percent": round((healthy_vegetation / total_count) * 100, 2),
            "stressed_vegetation_percent": round((stressed_vegetation / total_count) * 100, 2),
            "overall_health_score": round(health_score, 2),
            "health_category": _get_health_category(health_score)
        }
        
    elif index_type == "ndmi":
        # NDMI thresholds for moisture classification
        thresholds = {
            "very_dry": (-1.0, -0.4),
            "dry": (-0.4, -0.1),
            "moderate_moisture": (-0.1, 0.2),
            "well_hydrated": (0.2, 0.5),
            "very_wet": (0.5, 1.0)
        }
        
        # Optimized classification calculation using vectorized operations
        print(f"⚡ [FAST STATS] Computing NDMI classifications...")
        classifications = {}
        total_count = len(valid_data)
        
        for class_name, (min_val, max_val) in thresholds.items():
            # Use vectorized operations for speed
            mask = (valid_data >= min_val) & (valid_data < max_val)
            count = np.count_nonzero(mask)  # Faster than np.sum for boolean
            percentage = (count / total_count) * 100
            classifications[class_name] = {
                "percentage": round(percentage, 2),
                "pixel_count": int(count),
                "threshold": (min_val, max_val)
            }
        
        # Optimized moisture indicators calculation
        print(f"⚡ [FAST STATS] Computing NDMI moisture indicators...")
        well_hydrated = np.count_nonzero(valid_data >= 0.2)
        drought_stress = np.count_nonzero(valid_data < -0.1)
        moisture_score = (well_hydrated / total_count) * 100
        
        health_indicators = {
            "well_hydrated_percent": round((well_hydrated / total_count) * 100, 2),
            "drought_stress_percent": round((drought_stress / total_count) * 100, 2),
            "overall_moisture_score": round(moisture_score, 2),
            "moisture_category": _get_moisture_category(moisture_score)
        }
    
    # Fast spatial statistics (skip for very large images to avoid blocking)
    total_pixels = index_array.size
    if total_pixels > 20_000_000:  # Skip spatial analysis for images larger than 20 megapixels
        print(f"[STATISTICS] Skipping spatial analysis for large image ({total_pixels:,} pixels)")
        spatial_stats = {
            "spatial_heterogeneity": 0.0,
            "edge_density": 0.0,
            "high_value_patches": 0,
            "low_value_patches": 0,
            "avg_high_patch_size": 0.0,
            "avg_low_patch_size": 0.0,
            "patch_fragmentation": 0.0,
            "processing_note": "Skipped for performance (image too large)"
        }
    else:
        try:
            spatial_stats = _calculate_spatial_statistics(index_array, index_type)
        except Exception as e:
            print(f"[STATISTICS] Warning: Could not calculate spatial statistics: {e}")
            spatial_stats = {
                "spatial_heterogeneity": 0.0,
                "edge_density": 0.0,
                "high_value_patches": 0,
                "low_value_patches": 0,
                "avg_high_patch_size": 0.0,
                "avg_low_patch_size": 0.0,
                "patch_fragmentation": 0.0,
                "processing_note": "Failed due to error"
            }
    
    # Combine all statistics
    comprehensive_stats = {
        "basic_statistics": basic_stats,
        "percentiles": percentile_stats,
        "classifications": classifications,
        "health_indicators": health_indicators,
        "spatial_analysis": spatial_stats,
        "metadata": {
            "index_type": index_type,
            "total_pixels": int(index_array.size),
            "valid_pixels": len(valid_data),
            "invalid_pixels": int(index_array.size - len(valid_data)),
            "data_completeness": round((len(valid_data) / index_array.size) * 100, 2)
        }
    }
    
    print(f"[STATISTICS] Completed comprehensive statistics calculation for {index_type}")
    return comprehensive_stats

def _calculate_spatial_statistics(index_array: np.ndarray, index_type: str) -> Dict[str, Any]:
    """Calculate spatial pattern statistics with ultra-fast performance for large images"""
    print(f"⚡ [FAST SPATIAL] Calculating optimized spatial statistics for {index_type}...")
    
    # For large images, use downsampling to speed up spatial analysis
    h, w = index_array.shape
    total_pixels = h * w
    
    # Use downsampling for images larger than 1 megapixel to dramatically speed up processing
    if total_pixels > 1_000_000:  # 1 megapixel
        # Calculate downsampling factor to target ~500K pixels
        downsample_factor = max(2, int(np.sqrt(total_pixels / 500_000)))
        print(f"[FAST SPATIAL] Large image detected ({total_pixels:,} pixels), downsampling by factor {downsample_factor}")
        
        # Downsample using slicing (much faster than resize)
        downsampled = index_array[::downsample_factor, ::downsample_factor]
        spatial_array = downsampled
    else:
        spatial_array = index_array
    
    # Remove NaN values by replacing with median for spatial analysis (optimized)
    valid_median = np.nanmedian(spatial_array)
    spatial_array = np.where(np.isnan(spatial_array), valid_median, spatial_array)
    
    # Fast spatial heterogeneity (coefficient of variation)
    mean_val = np.mean(spatial_array)
    spatial_heterogeneity = np.std(spatial_array) / np.abs(mean_val) if mean_val != 0 else 0
    
    # Fast edge detection using gradient approximation (much faster than Sobel)
    try:
        # Use simple gradient instead of expensive Sobel filter
        grad_x = np.abs(np.diff(spatial_array, axis=1))
        grad_y = np.abs(np.diff(spatial_array, axis=0))
        edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2
    except Exception:
        edge_density = 0.0
    
    # Fast clustering analysis using simple thresholding
    if index_type == "ndvi":
        high_value_mask = spatial_array > 0.5  # Dense vegetation
        low_value_mask = spatial_array < 0.2   # Sparse/no vegetation
    else:  # NDMI
        high_value_mask = spatial_array > 0.3  # Well hydrated
        low_value_mask = spatial_array < -0.2  # Dry areas
    
    # Simplified patch analysis for speed
    try:
        # Count transitions instead of full connected component analysis
        # This is much faster and gives similar insights
        
        high_pixels = np.sum(high_value_mask)
        low_pixels = np.sum(low_value_mask)
        
        # Estimate patch count using transition counting (much faster than measure.label)
        if high_pixels > 0:
            # Count horizontal and vertical transitions in high value areas
            high_transitions_h = np.sum(np.abs(np.diff(high_value_mask.astype(int), axis=1)))
            high_transitions_v = np.sum(np.abs(np.diff(high_value_mask.astype(int), axis=0)))
            high_patches = max(1, (high_transitions_h + high_transitions_v) // 4)  # Rough estimate
            avg_high_patch_size = high_pixels / high_patches
        else:
            high_patches = 0
            avg_high_patch_size = 0
            
        if low_pixels > 0:
            # Count horizontal and vertical transitions in low value areas
            low_transitions_h = np.sum(np.abs(np.diff(low_value_mask.astype(int), axis=1)))
            low_transitions_v = np.sum(np.abs(np.diff(low_value_mask.astype(int), axis=0)))
            low_patches = max(1, (low_transitions_h + low_transitions_v) // 4)  # Rough estimate
            avg_low_patch_size = low_pixels / low_patches
        else:
            low_patches = 0
            avg_low_patch_size = 0
            
    except Exception as e:
        print(f"[SPATIAL] Fast patch analysis failed: {e}")
        high_patches = low_patches = 0
        avg_high_patch_size = avg_low_patch_size = 0
    
    spatial_stats = {
        "spatial_heterogeneity": round(float(spatial_heterogeneity), 4),
        "edge_density": round(float(edge_density), 4),
        "high_value_patches": int(high_patches),
        "low_value_patches": int(low_patches),
        "avg_high_patch_size": round(float(avg_high_patch_size), 2),
        "avg_low_patch_size": round(float(avg_low_patch_size), 2),
        "patch_fragmentation": round(float(high_patches + low_patches) / max(1, spatial_array.size / 1000), 4),
        "processing_note": f"Analyzed on {'downsampled' if total_pixels > 1_000_000 else 'original'} data"
    }
    
    print(f"[SPATIAL] Fast spatial statistics calculated: {spatial_stats}")
    return spatial_stats

def _get_health_category(health_score: float) -> str:
    """Categorize vegetation health based on score"""
    if health_score >= 70:
        return "Excellent"
    elif health_score >= 50:
        return "Good"
    elif health_score >= 30:
        return "Moderate"
    elif health_score >= 15:
        return "Poor"
    else:
        return "Very Poor"

def _get_moisture_category(moisture_score: float) -> str:
    """Categorize moisture levels based on score"""
    if moisture_score >= 70:
        return "Well Hydrated"
    elif moisture_score >= 50:
        return "Adequate Moisture"
    elif moisture_score >= 30:
        return "Moderate Moisture"
    elif moisture_score >= 15:
        return "Low Moisture"
    else:
        return "Drought Stress"

# Enhanced visualization function removed - we only create colored images now and store stats in DB

def _prepare_llm_summary(statistics: Dict[str, Any], index_type: str) -> str:
    """Prepare a concise, LLM-friendly summary of the statistics"""
    if not statistics or "basic_statistics" not in statistics:
        return f"No {index_type.upper()} statistics available"
    
    basic = statistics.get("basic_statistics", {})
    classifications = statistics.get("classifications", {})
    health = statistics.get("health_indicators", {})
    spatial = statistics.get("spatial_analysis", {})
    
    if index_type == "ndvi":
        summary = f"""NDVI Analysis Summary:
• Average NDVI: {basic.get('mean', 0):.3f} (Range: {basic.get('min', 0):.3f} to {basic.get('max', 0):.3f})
• Vegetation Health: {health.get('overall_health_score', 0):.1f}% ({health.get('health_category', 'Unknown')})
• Dense Vegetation: {classifications.get('dense_vegetation', {}).get('percentage', 0):.1f}%
• Moderate Vegetation: {classifications.get('moderate_vegetation', {}).get('percentage', 0):.1f}%
• Sparse/Bare Areas: {classifications.get('sparse_vegetation', {}).get('percentage', 0) + classifications.get('bare_soil', {}).get('percentage', 0):.1f}%
• Spatial Fragmentation: {spatial.get('patch_fragmentation', 0):.3f}
• Data Quality: {statistics.get('metadata', {}).get('data_completeness', 0):.1f}% complete"""
    
    else:  # NDMI
        summary = f"""NDMI Analysis Summary:
• Average NDMI: {basic.get('mean', 0):.3f} (Range: {basic.get('min', 0):.3f} to {basic.get('max', 0):.3f})
• Moisture Level: {health.get('overall_moisture_score', 0):.1f}% ({health.get('moisture_category', 'Unknown')})
• Well Hydrated Areas: {classifications.get('well_hydrated', {}).get('percentage', 0):.1f}%
• Moderate Moisture: {classifications.get('moderate_moisture', {}).get('percentage', 0):.1f}%
• Dry/Stressed Areas: {classifications.get('dry', {}).get('percentage', 0) + classifications.get('very_dry', {}).get('percentage', 0):.1f}%
• Spatial Heterogeneity: {spatial.get('spatial_heterogeneity', 0):.3f}
• Data Quality: {statistics.get('metadata', {}).get('data_completeness', 0):.1f}% complete"""
    
    return summary

async def process_image_for_indices(
    db_file: Any, file_path: Path, db: AsyncSession
) -> Dict[str, Any]:
    """
    Process NDVI and NDMI for the given image file, save outputs and thumbnails.
    Returns dict with success status and details.
    """
    import os
    import uuid
    import datetime
    try:
        print(f"[NDVI/NDMI] Starting processing for: {file_path}")
        ndvi_dir = file_path.parent / "ndvi"
        ndmi_dir = file_path.parent / "ndmi"
        ndvi_thumb_dir = ndvi_dir / "thumbnails"
        ndmi_thumb_dir = ndmi_dir / "thumbnails"
        ndvi_dir.mkdir(exist_ok=True)
        ndmi_dir.mkdir(exist_ok=True)
        ndvi_thumb_dir.mkdir(exist_ok=True)
        ndmi_thumb_dir.mkdir(exist_ok=True)

        ndvi_path = ndvi_dir / f"{file_path.stem}_ndvi.tif"
        ndmi_path = ndmi_dir / f"{file_path.stem}_ndmi.tif"

        ndvi_result = await _calculate_index(
            file_path, ndvi_path, index_type="ndvi"
        )
        print(f"[NDVI/NDMI] NDVI result: {ndvi_result}")
        ndmi_result = await _calculate_index(
            file_path, ndmi_path, index_type="ndmi"
        )
        print(f"[NDVI/NDMI] NDMI result: {ndmi_result}")

        # Generate thumbnails for NDVI
        ndvi_thumb_results = _generate_thumbnails(ndvi_path, ndvi_thumb_dir)
        ndmi_thumb_results = _generate_thumbnails(ndmi_path, ndmi_thumb_dir)
        print(f"[NDVI/NDMI] NDVI thumbnails: {ndvi_thumb_results}")
        print(f"[NDVI/NDMI] NDMI thumbnails: {ndmi_thumb_results}")

        # --- DB: Save NDVI/NDMI as ProcessedImage records ---
        async def save_processed_image_record(index_type: str, result_data: dict, thumb_results: dict):
            if not result_data.get("success"):
                print(f"[NDVI/NDMI] Skipping DB save for {index_type} - processing failed")
                return
                
            # Now we only have colored_path since we removed original_path
            colored_path = Path(result_data["colored_path"])
            
            result = await db.execute(
                select(ProcessedImage).where(
                    ProcessedImage.file_id == db_file.id,
                    ProcessedImage.processed_image_type == ProcessedImageType(index_type.lower())
                )
            )
            existing = result.scalar_one_or_none()
            
            if not existing and colored_path.exists():
                now = datetime.datetime.now(datetime.timezone.utc)
                
                # Get image dimensions from colored file
                width = height = None
                try:
                    with rasterio.open(colored_path) as src:
                        width = src.width
                        height = src.height
                except Exception as e:
                    print(f"[NDVI/NDMI] Error reading processed image dimensions: {e}")
                
                # Prepare comprehensive processing statistics for database storage
                processing_stats = {
                    # Original statistics from calculation
                    "statistics": result_data.get("statistics", {}),
                    "processing_info": result_data.get("processing_info", {}),
                    
                    # File path for colored image only
                    "file_paths": {
                        "colored_tif": str(colored_path)  # Store the colored image path
                    },
                    
                    # Thumbnail information
                    "thumbnails": thumb_results,
                    
                    # Processing metadata
                    "processing_metadata": {
                        "processing_version": "production_v1.0",
                        "color_mapping_applied": result_data.get("processing_info", {}).get("color_mapping_applied", False),
                        "statistics_calculated": result_data.get("processing_info", {}).get("statistics_calculated", False),
                        "bands_used": result_data.get("processing_info", {}).get("bands_used"),
                        "calculation_method": result_data.get("processing_info", {}).get("calculation_method")
                    },
                    
                    # Quick access statistics for LLM
                    "summary_for_llm": _prepare_llm_summary(result_data.get("statistics", {}), index_type)
                }
                
                new_img = ProcessedImage(
                    id=uuid.uuid4(),
                    file_id=db_file.id,
                    processed_image_type=ProcessedImageType(index_type.lower()),
                    processed_image_path=str(colored_path),
                    original_filename=db_file.original_filename,
                    processed_filename=colored_path.name,
                    width=width,
                    height=height,
                    file_size=os.path.getsize(colored_path),
                    processing_stats=processing_stats,
                    processing_status="completed",
                    created_at=now,
                    updated_at=now
                )
                db.add(new_img)
                await db.commit()
                print(f"[NDVI/NDMI] Created enhanced ProcessedImage record for {index_type}: {colored_path}")
                print(f"[NDVI/NDMI] Stored comprehensive statistics with {len(processing_stats.get('statistics', {}))} categories")
            else:
                print(f"[NDVI/NDMI] ProcessedImage record for {index_type} already exists or file missing: {colored_path}")

        # Save enhanced results to database
        if ndvi_result.get("success") and Path(ndvi_result["colored_path"]).exists():
            await save_processed_image_record("ndvi", ndvi_result, ndvi_thumb_results)
        if ndmi_result.get("success") and Path(ndmi_result["colored_path"]).exists():
            await save_processed_image_record("ndmi", ndmi_result, ndmi_thumb_results)

        return {
            "success": True,
            "processing_version": "production_v1.0",
            "enhancements_applied": {
                "color_mapping": True,
                "comprehensive_statistics": True,
                "spatial_analysis": True
            },
            "ndvi": {
                "success": ndvi_result.get("success", False),
                "colored_path": ndvi_result.get("colored_path") if ndvi_result.get("success") else None,
                "thumbnails": ndvi_thumb_results,
                "statistics": ndvi_result.get("statistics") if ndvi_result.get("success") else None,
                "llm_summary": _prepare_llm_summary(ndvi_result.get("statistics", {}), "ndvi") if ndvi_result.get("success") else None
            },
            "ndmi": {
                "success": ndmi_result.get("success", False),
                "colored_path": ndmi_result.get("colored_path") if ndmi_result.get("success") else None,
                "thumbnails": ndmi_thumb_results,
                "statistics": ndmi_result.get("statistics") if ndmi_result.get("success") else None,
                "llm_summary": _prepare_llm_summary(ndmi_result.get("statistics", {}), "ndmi") if ndmi_result.get("success") else None
            }
        }
    except Exception as e:
        print(f"[NDVI/NDMI] Error processing NDVI/NDMI for {file_path}: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error processing NDVI/NDMI for {file_path}: {e}")
        return {"success": False, "reason": str(e)}

async def _calculate_index(
    input_path: Path, output_path: Path, index_type: str
) -> Dict[str, Any]:
    try:
        print(f"[NDVI/NDMI] Calculating {index_type} for {input_path} -> {output_path}")
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            # Always output GeoTIFF for NDVI/NDMI
            profile.update(
                driver="GTiff",
                dtype=rasterio.float32,
                count=1,
                compress="lzw"
            )
            # Remove any photometric/tiled/blockxsize/blockysize for float32 single-band
            profile.pop("photometric", None)
            profile.pop("tiled", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)

            height, width = src.height, src.width
            window = Window(0, 0, width, height)
            arr = src.read(window=window)
            print(f"[NDVI/NDMI] Array shape: {arr.shape}")
            
            # Band detection
            if arr.shape[0] >= 4:
                # Multispectral: [Red, Green, Blue, NIR, ...]
                red = arr[0].astype(np.float32)
                green = arr[1].astype(np.float32)
                blue = arr[2].astype(np.float32)
                nir = arr[3].astype(np.float32)
                swir = arr[4].astype(np.float32) if arr.shape[0] > 4 else None
                print(f"[NDVI/NDMI] Multispectral bands detected. R,G,B,NIR,SWIR: {red.shape}, {green.shape}, {blue.shape}, {nir.shape}, {swir.shape if swir is not None else 'None'}")
            elif arr.shape[0] == 3:
                # RGB fallback
                red = arr[0].astype(np.float32)
                green = arr[1].astype(np.float32)
                blue = arr[2].astype(np.float32)
                nir = None
                swir = None
                print(f"[NDVI/NDMI] RGB bands detected. R,G,B: {red.shape}, {green.shape}, {blue.shape}")
            else:
                print(f"[NDVI/NDMI] Unsupported band count: {arr.shape[0]}")
                raise ValueError("Unsupported band count for NDVI/NDMI calculation")

            if index_type == "ndvi":
                if nir is not None:
                    index = (nir - red) / (nir + red + 1e-6)
                    print(f"[NDVI/NDMI] NDVI calculated using NIR and Red")
                else:
                    # VARI fallback for RGB
                    index = (green - red) / (green + red - blue + 1e-6)
                    print(f"[NDVI/NDMI] NDVI fallback (VARI) using Green, Red, Blue")
            elif index_type == "ndmi":
                if nir is not None and swir is not None:
                    index = (nir - swir) / (nir + swir + 1e-6)
                    print(f"[NDVI/NDMI] NDMI calculated using NIR and SWIR")
                elif nir is not None:
                    # Moisture stress index fallback
                    index = (nir - red) / (nir + red + 1e-6)
                    print(f"[NDVI/NDMI] NDMI fallback (Moisture stress) using NIR and Red")
                else:
                    # NDRGBI fallback for RGB
                    index = (green - blue) / (green + blue + 1e-6)
                    print(f"[NDVI/NDMI] NDMI fallback (NDRGBI) using Green and Blue")
            else:
                print(f"[NDVI/NDMI] Unknown index type: {index_type}")
                raise ValueError(f"Unknown index type: {index_type}")

            # Clip values to valid range
            index = np.clip(index, -1.0, 1.0)
            print(f"[NDVI/NDMI] {index_type} min: {np.nanmin(index)}, max: {np.nanmax(index)}, mean: {np.nanmean(index)}")

            # === PRODUCTION-LEVEL ENHANCEMENTS ===
            
            # 1. Calculate comprehensive statistics
            print(f"[PRODUCTION] Calculating comprehensive statistics for {index_type}...")
            comprehensive_stats = _calculate_comprehensive_statistics(index, index_type)
            
            # 2. Apply color mapping for better LLM visualization
            print(f"[PRODUCTION] Applying color mapping for {index_type}...")
            colored_array = _apply_color_mapping(index, index_type)
            
            # 3. Normalize index values to 0-255 for better contrast in standard output
            print(f"[PRODUCTION] Normalizing {index_type} values...")
            normalized_index = np.clip((index + 1) * 127.5, 0, 255).astype(np.uint8)
            
            # Ensure output path ends with .tif and use colored naming  
            if not str(output_path).lower().endswith((".tif", ".tiff")):
                output_path = output_path.with_suffix('.tif')
            
            # Use colored naming directly for the main output
            colored_output_path = output_path.parent / f"{output_path.stem}_colored.tif"
            
            # Save ONLY the color-mapped version (no original float32 or enhanced viz)
            colored_profile = profile.copy()
            colored_profile.update(dtype=rasterio.uint8, count=3)  # RGB
            
            with rasterio.open(colored_output_path, "w", **colored_profile) as dst:
                # Write RGB bands
                for i in range(3):
                    dst.write(colored_array[:, :, i], i + 1)
                dst.update_tags(
                    index_type=f"{index_type}_colored",
                    source=str(input_path),
                    colormap_applied="true",
                    processing_version="production_v1.0",
                    statistics_available="true"
                )
            
            print(f"[PRODUCTION] Color-mapped {index_type} saved to {colored_output_path}")

            # Return simplified results with only colored image
            result = {
                "success": True,
                "index_type": index_type,
                "colored_path": str(colored_output_path),
                "statistics": comprehensive_stats,
                "processing_info": {
                    "bands_used": "NIR+Red" if nir is not None else "RGB_fallback",
                    "calculation_method": "standard" if (index_type == "ndvi" and nir is not None) or (index_type == "ndmi" and nir is not None and swir is not None) else "fallback",
                    "value_range": {"min": float(np.nanmin(index)), "max": float(np.nanmax(index))},
                    "normalized_range": {"min": 0, "max": 255},
                    "color_mapping_applied": True,
                    "statistics_calculated": True
                }
            }
            
            print(f"[PRODUCTION] {index_type} processing complete with enhanced features")
            return result

    except Exception as e:
        print(f"[NDVI/NDMI] Error calculating {index_type}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False, 
            "error": str(e),
            "index_type": index_type
        }

def _generate_thumbnails(tif_path: Path, thumb_dir: Path) -> Dict[str, str]:
    """Generate thumbnails for processed NDVI/NDMI images"""
    results = {}
    
    print(f"[THUMBNAIL] 🚀 Starting thumbnail generation for: {tif_path}")
    
    try:
        # Ensure thumbnail directory exists before proceeding
        thumb_dir.mkdir(parents=True, exist_ok=True)
        print(f"[THUMBNAIL] 📁 Created/verified thumbnail directory: {thumb_dir}")
        
        with rasterio.open(tif_path) as src:
            print(f"[THUMBNAIL] ✅ File is readable. Bands: {src.count}, Shape: {src.height}x{src.width}")
            
            arr = src.read(1, out_shape=(src.height, src.width), resampling=Resampling.bilinear)
            print(f"[THUMBNAIL] 📊 Array stats - Shape: {arr.shape}, Min: {np.min(arr):.3f}, Max: {np.max(arr):.3f}, Mean: {np.mean(arr):.3f}")
            
            # Map -1..1 to 0..255 for proper image display
            arr = np.clip((arr + 1) * 127.5, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
            print(f"[THUMBNAIL] ✅ Created PIL image: {img.size}")
            
            successful_thumbs = 0
            for size_name, width, height in THUMBNAIL_SIZES:
                try:
                    print(f"[THUMBNAIL] 🔄 Creating {size_name} thumbnail ({width}x{height})...")
                    
                    thumb = img.copy()
                    thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
                    
                    # Create final image with padding
                    final_img = Image.new("RGB", (width, height), (255, 255, 255))
                    paste_x = (width - thumb.width) // 2
                    paste_y = (height - thumb.height) // 2
                    final_img.paste(thumb, (paste_x, paste_y))
                    
                    # Save thumbnail
                    thumb_path = thumb_dir / f"{tif_path.stem}_{size_name}.jpg"
                    final_img.save(thumb_path, "JPEG", quality=85, optimize=True)
                    
                    # Verify the thumbnail was created successfully
                    if thumb_path.exists():
                        results[size_name] = str(thumb_path)
                        successful_thumbs += 1
                        print(f"[THUMBNAIL] ✅ {size_name} thumbnail created: {thumb_path} ({final_img.size})")
                    else:
                        print(f"[THUMBNAIL] ❌ Failed to create {size_name} thumbnail: {thumb_path}")
                        
                except Exception as thumb_error:
                    print(f"[THUMBNAIL] ❌ Error creating {size_name} thumbnail: {thumb_error}")
                    logger.error(f"Error creating {size_name} thumbnail for {tif_path}: {thumb_error}")
            
            print(f"[THUMBNAIL] 🎉 Thumbnail generation complete. Created {successful_thumbs}/{len(THUMBNAIL_SIZES)} thumbnails")
            
            # Retry logic for failed thumbnails
            if successful_thumbs < len(THUMBNAIL_SIZES):
                print(f"[THUMBNAIL] ⚠️ Only {successful_thumbs}/{len(THUMBNAIL_SIZES)} thumbnails created successfully")
                
    except Exception as e:
        print(f"[THUMBNAIL] 💥 Critical error in thumbnail generation: {e}")
        logger.error(f"Error generating thumbnails for {tif_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return results

async def get_processed_images_for_file(file_id: str, db: AsyncSession) -> dict:
    """
    Return NDVI/NDMI processed image info and thumbnails for a file.
    Now queries the ProcessedImage table for processed images using file_id.
    """
    print(f"[PROCESSED SERVICE DEBUG] === GET_PROCESSED_IMAGES START ===")
    print(f"[PROCESSED SERVICE DEBUG] file_id: {file_id}")
    
    import os
    import datetime
    from pathlib import Path
    import rasterio
    from models.processed_image import ProcessedImage
    from sqlalchemy import select

    def get_thumbs_from_db(processed_path: str):
        print(f"[PROCESSED SERVICE DEBUG] Getting thumbnails for path: {processed_path}")
        thumbs = {}
        thumb_dir = Path(processed_path).parent / "thumbnails"
        stem = Path(processed_path).stem
        print(f"[PROCESSED SERVICE DEBUG] Thumbnail directory: {thumb_dir}")
        print(f"[PROCESSED SERVICE DEBUG] Stem: {stem}")
        
        for size in ["small", "medium", "large"]:
            thumb_path = thumb_dir / f"{stem}_{size}.jpg"
            print(f"[PROCESSED SERVICE DEBUG] Checking {size} thumbnail: {thumb_path}")
            if thumb_path.exists():
                thumbs[size] = str(thumb_path)
                print(f"[PROCESSED SERVICE DEBUG] ✅ Found {size} thumbnail")
            else:
                print(f"[PROCESSED SERVICE DEBUG] ❌ Missing {size} thumbnail")
        
        print(f"[PROCESSED SERVICE DEBUG] Final thumbnails: {thumbs}")
        return thumbs

    def build_processed_image(img: ProcessedImage):
        print(f"[PROCESSED SERVICE DEBUG] Building processed image for: {img.id}")
        result = {
            "id": str(img.id),
            "path": img.processed_image_path,
            "filename": img.processed_filename,
            "width": img.width,
            "height": img.height,
            "file_size": img.file_size,
            "file_size_formatted": img.file_size_formatted,
            "created_at": img.created_at.isoformat() if img.created_at else None,
            "stats": img.processing_stats,
            "thumbnails": get_thumbs_from_db(img.processed_image_path)
        }
        print(f"[PROCESSED SERVICE DEBUG] Built image result: {result}")
        return result

    # Query for processed NDVI/NDMI images using the enum
    print(f"[PROCESSED SERVICE DEBUG] Querying for NDVI image...")
    ndvi_db = (await db.execute(
        select(ProcessedImage).where(
            ProcessedImage.file_id == file_id,
            ProcessedImage.processed_image_type == ProcessedImageType.NDVI
        )
    )).scalar_one_or_none()
    
    print(f"[PROCESSED SERVICE DEBUG] Querying for NDMI image...")
    ndmi_db = (await db.execute(
        select(ProcessedImage).where(
            ProcessedImage.file_id == file_id,
            ProcessedImage.processed_image_type == ProcessedImageType.NDMI
        )
    )).scalar_one_or_none()

    print(f"[PROCESSED SERVICE DEBUG] NDVI DB result: {ndvi_db}")
    print(f"[PROCESSED SERVICE DEBUG] NDMI DB result: {ndmi_db}")

    result = {"ndvi": None, "ndmi": None}
    
    if ndvi_db:
        print(f"[PROCESSED SERVICE DEBUG] Checking NDVI file exists: {ndvi_db.processed_image_path}")
        if Path(ndvi_db.processed_image_path).exists():
            print(f"[PROCESSED SERVICE DEBUG] ✅ NDVI file exists, building result")
            result["ndvi"] = build_processed_image(ndvi_db)
        else:
            print(f"[PROCESSED SERVICE DEBUG] ❌ NDVI file missing on disk")
    else:
        print(f"[PROCESSED SERVICE DEBUG] ❌ No NDVI record in database")
        
    if ndmi_db:
        print(f"[PROCESSED SERVICE DEBUG] Checking NDMI file exists: {ndmi_db.processed_image_path}")
        if Path(ndmi_db.processed_image_path).exists():
            print(f"[PROCESSED SERVICE DEBUG] ✅ NDMI file exists, building result")
            result["ndmi"] = build_processed_image(ndmi_db)
        else:
            print(f"[PROCESSED SERVICE DEBUG] ❌ NDMI file missing on disk")
    else:
        print(f"[PROCESSED SERVICE DEBUG] ❌ No NDMI record in database")

    # Fallback: legacy path-based lookup for backward compatibility
    if not result["ndvi"] or not result["ndmi"]:
        print(f"[PROCESSED SERVICE DEBUG] Some images missing, checking legacy paths...")
        # ... (keep legacy path-based code here if needed) ...
        pass

    if not result["ndvi"] and not result["ndmi"]:
        print(f"[PROCESSED SERVICE DEBUG] ❌ No NDVI/NDMI outputs found for file_id={file_id}")
        return {"error": f"No NDVI/NDMI outputs found for file_id={file_id}."}
    
    print(f"[PROCESSED SERVICE DEBUG] ✅ Final result: {result}")
    print(f"[PROCESSED SERVICE DEBUG] === GET_PROCESSED_IMAGES END ===")
    return result 

# === ULTRA-FAST STREAMLINED PROCESSING PIPELINE ===

async def process_image_for_indices_fast(
    db_file: Any, file_path: Path, db: AsyncSession
) -> Dict[str, Any]:
    """
    🚀 ULTRA-FAST streamlined processing pipeline for NDVI and NDMI.
    
    Optimizations:
    - Direct RGB processing (no file I/O for calculations)
    - In-memory color mapping
    - Skip intermediate TIF files
    - Parallel processing
    - Direct thumbnail generation from colored arrays
    - Single-pass statistics calculation
    """
    import os
    import uuid
    import datetime
    
    start_time = time.time()
    print(f"🚀 [ULTRA-FAST] Starting streamlined processing for: {file_path}")
    
    try:
        # === STEP 1: Setup directories ===
        setup_start = time.time()
        ndvi_dir = file_path.parent / "ndvi"
        ndmi_dir = file_path.parent / "ndmi"
        ndvi_thumb_dir = ndvi_dir / "thumbnails"
        ndmi_thumb_dir = ndmi_dir / "thumbnails"
        
        # Create all directories at once
        for dir_path in [ndvi_dir, ndmi_dir, ndvi_thumb_dir, ndmi_thumb_dir]:
            dir_path.mkdir(exist_ok=True)
        
        setup_time = time.time() - setup_start
        print(f"⚡ [ULTRA-FAST] Directory setup: {setup_time:.2f}s")
        
        # === STEP 2: Load and process image in one pass ===
        calc_start = time.time()
        
        # Read image data once
        with rasterio.open(file_path) as src:
            profile = src.profile.copy()
            arr = src.read()  # Read all bands
            print(f"⚡ [ULTRA-FAST] Loaded image: {arr.shape}")
        
        # Band detection and extraction
        if arr.shape[0] >= 4:
            # Multispectral: [Red, Green, Blue, NIR, ...]
            red = arr[0].astype(np.float32)
            green = arr[1].astype(np.float32) 
            blue = arr[2].astype(np.float32)
            nir = arr[3].astype(np.float32)
            bands_used = "NIR+Red"
        elif arr.shape[0] == 3:
            # RGB fallback
            red = arr[0].astype(np.float32)
            green = arr[1].astype(np.float32)
            blue = arr[2].astype(np.float32)
            nir = None
            bands_used = "RGB_fallback"
        else:
            raise ValueError(f"Unsupported band count: {arr.shape[0]}")
        
        # === STEP 3: Calculate indices in parallel ===
        
        # NDVI calculation
        if nir is not None:
            ndvi = (nir - red) / (nir + red + 1e-6)
            print(f"[ULTRA-FAST] NDVI using NIR and Red")
        else:
            ndvi = (green - red) / (green + red - blue + 1e-6)
            print(f"[ULTRA-FAST] NDVI using VARI (RGB fallback)")
        ndvi = np.clip(ndvi, -1.0, 1.0)
        
        # NDMI calculation  
        if nir is not None:
            ndmi = (nir - red) / (nir + red + 1e-6)  # Moisture stress index
            print(f"[ULTRA-FAST] NDMI using moisture stress index")
        else:
            ndmi = (green - blue) / (green + blue + 1e-6)  # RGB fallback
            print(f"[ULTRA-FAST] NDMI using RGB fallback")
        ndmi = np.clip(ndmi, -1.0, 1.0)
        
        print(f"[ULTRA-FAST] NDVI: min={np.nanmin(ndvi):.3f}, max={np.nanmax(ndvi):.3f}, mean={np.nanmean(ndvi):.3f}")
        print(f"[ULTRA-FAST] NDMI: min={np.nanmin(ndmi):.3f}, max={np.nanmax(ndmi):.3f}, mean={np.nanmean(ndmi):.3f}")
        
        calc_time = time.time() - calc_start
        print(f"⚡ [ULTRA-FAST] Index calculations: {calc_time:.2f}s")
        
        # === STEP 4: Statistics and color mapping in parallel ===
        stats_start = time.time()
        
        # Calculate statistics in parallel
        ndvi_stats_future = asyncio.create_task(asyncio.to_thread(_calculate_comprehensive_statistics, ndvi, "ndvi"))
        ndmi_stats_future = asyncio.create_task(asyncio.to_thread(_calculate_comprehensive_statistics, ndmi, "ndmi"))
        
        # Apply color mapping in parallel
        ndvi_colored_future = asyncio.create_task(asyncio.to_thread(_apply_color_mapping, ndvi, "ndvi"))
        ndmi_colored_future = asyncio.create_task(asyncio.to_thread(_apply_color_mapping, ndmi, "ndmi"))
        
        # Wait for all parallel operations
        ndvi_stats, ndmi_stats, ndvi_colored, ndmi_colored = await asyncio.gather(
            ndvi_stats_future, ndmi_stats_future, ndvi_colored_future, ndmi_colored_future
        )
        
        stats_time = time.time() - stats_start
        print(f"⚡ [ULTRA-FAST] Statistics and color mapping: {stats_time:.2f}s")
        
        # === STEP 5: Save colored images and generate thumbnails ===
        save_start = time.time()
        
        # Prepare file paths
        ndvi_colored_path = ndvi_dir / f"{file_path.stem}_ndvi_colored.tif"
        ndmi_colored_path = ndmi_dir / f"{file_path.stem}_ndmi_colored.tif"
        
        # Prepare profile for colored images
        colored_profile = profile.copy()
        colored_profile.update({
            "driver": "GTiff",
            "dtype": rasterio.uint8,
            "count": 3,
            "compress": "lzw"
        })
        
        # Remove problematic keys
        for key in ["photometric", "tiled", "blockxsize", "blockysize"]:
            colored_profile.pop(key, None)
        
        # Save colored images and generate thumbnails in parallel
        async def save_and_thumbnail(colored_array, output_path, thumb_dir, index_type):
            # Save colored image
            with rasterio.open(output_path, "w", **colored_profile) as dst:
                for i in range(3):
                    dst.write(colored_array[:, :, i], i + 1)
                dst.update_tags(
                    index_type=f"{index_type}_colored",
                    source=str(file_path),
                    processing_version="ultra_fast_v4.0",
                    colormap_applied="true"
                )
            
            # Generate thumbnails directly from colored array
            thumb_results = {}
            try:
                # Convert array to PIL Image
                img = Image.fromarray(colored_array, mode="RGB")
                
                for size_name, width, height in THUMBNAIL_SIZES:
                    thumb = img.copy()
                    thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
                    
                    # Create final image with padding
                    final_img = Image.new("RGB", (width, height), (255, 255, 255))
                    paste_x = (width - thumb.width) // 2
                    paste_y = (height - thumb.height) // 2
                    final_img.paste(thumb, (paste_x, paste_y))
                    
                    # Save thumbnail
                    thumb_path = thumb_dir / f"{output_path.stem}_{size_name}.jpg"
                    final_img.save(thumb_path, "JPEG", quality=85, optimize=True)
                    
                    if thumb_path.exists():
                        thumb_results[size_name] = str(thumb_path)
                        print(f"[ULTRA-FAST] Created {size_name} thumbnail for {index_type}")
                
            except Exception as e:
                print(f"[ULTRA-FAST] Thumbnail error for {index_type}: {e}")
            
            return thumb_results
        
        # Execute saving and thumbnailing in parallel
        ndvi_task = save_and_thumbnail(ndvi_colored, ndvi_colored_path, ndvi_thumb_dir, "ndvi")
        ndmi_task = save_and_thumbnail(ndmi_colored, ndmi_colored_path, ndmi_thumb_dir, "ndmi")
        
        ndvi_thumb_results, ndmi_thumb_results = await asyncio.gather(ndvi_task, ndmi_task)
        
        save_time = time.time() - save_start
        print(f"⚡ [ULTRA-FAST] Save and thumbnails: {save_time:.2f}s")
        
        # === STEP 6: Database operations ===
        db_start = time.time()
        
        # Prepare results
        ndvi_result = {
            "success": True,
            "colored_path": str(ndvi_colored_path),
            "statistics": ndvi_stats,
            "processing_info": {
                "bands_used": bands_used,
                "calculation_method": "ultra_fast_v4.0",
                "color_mapping_applied": True,
                "statistics_calculated": True
            }
        }
        
        ndmi_result = {
            "success": True,
            "colored_path": str(ndmi_colored_path),
            "statistics": ndmi_stats,
            "processing_info": {
                "bands_used": bands_used,
                "calculation_method": "ultra_fast_v4.0",
                "color_mapping_applied": True,
                "statistics_calculated": True
            }
        }
        
        # Save to database
        await _save_processed_images_batch(
            db_file, ndvi_result, ndmi_result,
            ndvi_thumb_results, ndmi_thumb_results, db
        )
        
        db_time = time.time() - db_start
        print(f"⚡ [ULTRA-FAST] Database operations: {db_time:.2f}s")
        
        # === STEP 7: Final summary ===
        total_time = time.time() - start_time
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        processing_speed = file_size_mb / total_time if total_time > 0 else 0
        
        print(f"🏁 [ULTRA-FAST] Processing complete!")
        print(f"📊 [ULTRA-FAST] Total time: {total_time:.2f}s for {file_size_mb:.1f}MB")
        print(f"🚀 [ULTRA-FAST] Processing speed: {processing_speed:.1f}MB/s")
        print(f"⚡ [ULTRA-FAST] Breakdown: Setup={setup_time:.2f}s, Calc={calc_time:.2f}s, Stats={stats_time:.2f}s, Save={save_time:.2f}s, DB={db_time:.2f}s")
        
        return {
            "success": True,
            "processing_version": "ultra_fast_v4.0",
            "performance_metrics": {
                "total_time": total_time,
                "processing_speed_mbps": processing_speed,
                "breakdown": {
                    "setup_time": setup_time,
                    "calculation_time": calc_time,
                    "statistics_time": stats_time,
                    "save_time": save_time,
                    "database_time": db_time
                }
            },
            "enhancements_applied": {
                "streamlined_processing": True,
                "parallel_operations": True,
                "memory_efficient": True,
                "direct_thumbnails": True
            },
            "ndvi": {
                "success": ndvi_result.get("success", False),
                "colored_path": ndvi_result.get("colored_path"),
                "thumbnails": ndvi_thumb_results,
                "statistics": ndvi_stats,
                "llm_summary": _prepare_llm_summary(ndvi_stats, "ndvi")
            },
            "ndmi": {
                "success": ndmi_result.get("success", False),
                "colored_path": ndmi_result.get("colored_path"),
                "thumbnails": ndmi_thumb_results,
                "statistics": ndmi_stats,
                "llm_summary": _prepare_llm_summary(ndmi_stats, "ndmi")
            }
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ [ULTRA-FAST] Error after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "processing_time": total_time
        }

async def _save_processed_images_batch(
    db_file: Any,
    ndvi_result: Dict[str, Any], ndmi_result: Dict[str, Any],
    ndvi_thumb_results: Dict[str, str], ndmi_thumb_results: Dict[str, str],
    db: AsyncSession
):
    """
    Batch save NDVI and NDMI processed images to the database.
    """
    import os
    import uuid
    import datetime
    
    # NDVI
    if ndvi_result.get("success") and Path(ndvi_result["colored_path"]).exists():
        colored_path = ndvi_result["colored_path"]
        colored_path_obj = Path(colored_path)
        
        result = await db.execute(
            select(ProcessedImage).where(
                ProcessedImage.file_id == db_file.id,
                ProcessedImage.processed_image_type == ProcessedImageType.NDVI
            )
        )
        existing_ndvi = result.scalar_one_or_none()
        
        if not existing_ndvi:
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # Get image dimensions from colored file
            width = height = None
            try:
                with rasterio.open(colored_path) as src:
                    width = src.width
                    height = src.height
            except Exception as e:
                print(f"[FAST PIPELINE] Error reading NDVI processed image dimensions: {e}")
            
            # Prepare comprehensive processing statistics for database storage
            processing_stats = {
                "statistics": ndvi_result.get("statistics", {}),
                "processing_info": ndvi_result.get("processing_info", {}),
                "file_paths": {
                    "colored_tif": str(colored_path)
                },
                "thumbnails": ndvi_thumb_results,
                "processing_metadata": {
                    "processing_version": "fast_parallel_v2.0",
                    "color_mapping_applied": ndvi_result.get("processing_info", {}).get("color_mapping_applied", False),
                    "statistics_calculated": ndvi_result.get("processing_info", {}).get("statistics_calculated", False),
                    "bands_used": ndvi_result.get("processing_info", {}).get("bands_used"),
                    "calculation_method": ndvi_result.get("processing_info", {}).get("calculation_method")
                },
                "summary_for_llm": _prepare_llm_summary(ndvi_result.get("statistics", {}), "ndvi")
            }
            
            new_ndvi_img = ProcessedImage(
                id=uuid.uuid4(),
                file_id=db_file.id,
                processed_image_type=ProcessedImageType.NDVI,
                processed_image_path=str(colored_path),
                original_filename=db_file.original_filename,
                processed_filename=colored_path_obj.name,
                width=width,
                height=height,
                file_size=os.path.getsize(colored_path),
                processing_stats=processing_stats,
                processing_status="completed",
                created_at=now,
                updated_at=now
            )
            db.add(new_ndvi_img)
            print(f"[FAST PIPELINE] Created NDVI ProcessedImage record: {colored_path}")
        else:
            print(f"[FAST PIPELINE] NDVI ProcessedImage record already exists: {colored_path}")
    
    # NDMI
    if ndmi_result.get("success") and Path(ndmi_result["colored_path"]).exists():
        colored_path = ndmi_result["colored_path"]
        colored_path_obj = Path(colored_path)
        
        result = await db.execute(
            select(ProcessedImage).where(
                ProcessedImage.file_id == db_file.id,
                ProcessedImage.processed_image_type == ProcessedImageType.NDMI
            )
        )
        existing_ndmi = result.scalar_one_or_none()
        
        if not existing_ndmi:
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # Get image dimensions from colored file
            width = height = None
            try:
                with rasterio.open(colored_path) as src:
                    width = src.width
                    height = src.height
            except Exception as e:
                print(f"[FAST PIPELINE] Error reading NDMI processed image dimensions: {e}")
            
            # Prepare comprehensive processing statistics for database storage
            processing_stats = {
                "statistics": ndmi_result.get("statistics", {}),
                "processing_info": ndmi_result.get("processing_info", {}),
                "file_paths": {
                    "colored_tif": str(colored_path)
                },
                "thumbnails": ndmi_thumb_results,
                "processing_metadata": {
                    "processing_version": "fast_parallel_v2.0",
                    "color_mapping_applied": ndmi_result.get("processing_info", {}).get("color_mapping_applied", False),
                    "statistics_calculated": ndmi_result.get("processing_info", {}).get("statistics_calculated", False),
                    "bands_used": ndmi_result.get("processing_info", {}).get("bands_used"),
                    "calculation_method": ndmi_result.get("processing_info", {}).get("calculation_method")
                },
                "summary_for_llm": _prepare_llm_summary(ndmi_result.get("statistics", {}), "ndmi")
            }
            
            new_ndmi_img = ProcessedImage(
                id=uuid.uuid4(),
                file_id=db_file.id,
                processed_image_type=ProcessedImageType.NDMI,
                processed_image_path=str(colored_path),
                original_filename=db_file.original_filename,
                processed_filename=colored_path_obj.name,
                width=width,
                height=height,
                file_size=os.path.getsize(colored_path),
                processing_stats=processing_stats,
                processing_status="completed",
                created_at=now,
                updated_at=now
            )
            db.add(new_ndmi_img)
            print(f"[FAST PIPELINE] Created NDMI ProcessedImage record: {colored_path}")
        else:
            print(f"[FAST PIPELINE] NDMI ProcessedImage record already exists: {colored_path}")
    
    await db.commit()
    print(f"[FAST PIPELINE] Batch DB operations completed. Committed changes.")

async def _initialize_output_files(
    input_path: Path, ndvi_path: Path, ndmi_path: Path, 
    width: int, height: int, profile: Dict
):
    """Initialize output files for chunked processing"""
    print(f"🔧 [CHUNKED] Initializing output files...")
    
    def init_file(output_path: Path, index_type: str):
        # Create profile for colored output
        output_profile = profile.copy()
        output_profile.update({
            "driver": "GTiff",
            "dtype": rasterio.uint8,
            "count": 3,  # RGB
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })
        
        # Remove problematic keys
        for key in ["photometric"]:
            output_profile.pop(key, None)
        
        # Initialize with zeros
        with rasterio.open(output_path, "w", **output_profile) as dst:
            zero_chunk = np.zeros((512, 512), dtype=np.uint8)
            
            for band in range(3):
                for y in range(0, height, 512):
                    for x in range(0, width, 512):
                        window = Window(x, y, min(512, width - x), min(512, height - y))
                        chunk_data = zero_chunk[:window.height, :window.width]
                        dst.write(chunk_data, band + 1, window=window)
            
            dst.update_tags(
                index_type=f"{index_type}_colored",
                source=str(input_path),
                processing_version="ultra_fast_chunked_v3.0",
                colormap_applied="true"
            )
    
    # Initialize both files in parallel
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, init_file, ndvi_path, "ndvi"),
            loop.run_in_executor(executor, init_file, ndmi_path, "ndmi")
        ]
        await asyncio.gather(*tasks)

# Chunked processing functions removed - using ultra-fast streamlined processing instead