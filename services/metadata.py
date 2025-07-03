#!/usr/bin/env python3
"""
Production-Level Image Metadata Extractor
Extracts comprehensive metadata and stores in FileMetadata model

Integrates with FastAPI backend to extract and store metadata
for uploaded files in the database.
"""

import sys
import os
import json
import math
import xml.etree.ElementTree as ET
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from sqlalchemy.ext.asyncio import AsyncSession

# Import database models
from models.file_metadata import FileMetadata
from models.file import File

class ProductionMetadataExtractor:
    """Production-level metadata extractor for any type of images"""
    
    def __init__(self):
        self.supported_formats = {
            # Standard image formats
            '.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif',
            # Geospatial image formats
            '.jp2', '.j2k', '.ecw', '.sid', '.vrt',
            # Raw image formats
            '.raw', '.cr2', '.nef', '.arw', '.dng',
            # Other image formats
            '.tga', '.ico', '.svg'
        }
        self.world_file_extensions = {
            '.jpg': '.jgw', '.jpeg': '.jgw', 
            '.tif': '.tfw', '.tiff': '.tfw', 
            '.png': '.pgw', '.bmp': '.bpw',
            '.jp2': '.j2w', '.j2k': '.j2w',
            '.ecw': '.eww', '.sid': '.sdw'
        }
    
    async def extract_and_save_metadata(self, file_path: str, file_id: str, db: AsyncSession) -> bool:
        """
        Extract metadata and save to FileMetadata model (Async)
        
        Args:
            file_path: Path to image file
            file_id: ID of the File record
            db: Database session
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract metadata using async method
            metadata = await self.extract_metadata_async(file_path)
            
            if metadata["processing_status"] == "error":
                return False
            
            # Map extracted metadata to FileMetadata fields
            gps_data = metadata.get("gps_data", {})
            camera_settings = metadata.get("camera_settings", {})
            flight_params = metadata.get("flight_parameters", {})
            basic_info = metadata.get("basic_info", {})
            
            # Helper function to safely convert to float
            def safe_float(value):
                """Safely convert value to float"""
                if value is None:
                    return None
                try:
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        return float(value[0])
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Helper function to safely convert to int
            def safe_int(value):
                """Safely convert value to integer"""
                if value is None:
                    return None
                try:
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        return int(value[0])
                    return int(value)
                except (ValueError, TypeError):
                    return None
            
            # Helper function to safely convert to string
            def safe_str(value):
                """Safely convert value to string"""
                if value is None:
                    return None
                try:
                    return str(value)
                except (ValueError, TypeError):
                    return None
            
            # Create FileMetadata record with proper field mapping and type conversion
            file_metadata = FileMetadata(
                file_id=file_id,
                # Camera information
                camera_make=safe_str(flight_params.get("drone_manufacturer") or camera_settings.get("camera_make")),
                camera_model=safe_str(flight_params.get("drone_model") or camera_settings.get("camera_model")),
                lens_model=safe_str(camera_settings.get("lens_model")),
                
                # Technical settings - ensure proper type conversion
                focal_length=safe_float(camera_settings.get("focal_length_mm")),
                aperture=safe_float(camera_settings.get("aperture_f_number")),
                shutter_speed=safe_str(camera_settings.get("exposure_time_fraction")),
                iso_speed=safe_int(camera_settings.get("iso_speed")),
                
                # GPS data - ensure proper type conversion
                latitude=safe_float(gps_data.get("latitude")),
                longitude=safe_float(gps_data.get("longitude")),
                altitude=safe_float(gps_data.get("altitude_meters")),
                has_gps=gps_data.get("has_gps", False),
                
                # Date/time
                date_taken=self._parse_datetime(flight_params.get("capture_datetime")),
                
                # Image technical details
                color_space=safe_str(metadata.get("radiometric_data", {}).get("color_space")),
                white_balance=safe_str(metadata.get("radiometric_data", {}).get("white_balance")),
                flash_used=bool(camera_settings.get("flash_mode")) if camera_settings.get("flash_mode") else None,
                exposure_mode=safe_str(camera_settings.get("exposure_mode")),
                metering_mode=safe_str(camera_settings.get("metering_mode")),
                
                # Quality
                image_quality=safe_str(metadata.get("radiometric_data", {}).get("spectral_type")),
                bit_depth=self._extract_bit_depth_number(metadata.get("radiometric_data", {}).get("bit_depth")),
                
                # Professional detection
                is_professional_grade=metadata.get("contextual_data", {}).get("equipment_context", {}).get("is_professional_grade", False),
                equipment_category=safe_str(self._determine_equipment_category(flight_params.get("drone_model"))),
                
                # Raw metadata storage
                raw_exif_data=self._clean_exif_data(metadata.get("raw_exif", {})),
                processed_metadata=self._clean_value_for_json({
                    "extraction_info": metadata.get("extraction_info", {}),
                    "basic_info": basic_info,
                    "geometric_data": metadata.get("geometric_data", {}),
                    "elevation_data": metadata.get("elevation_data", {}),
                    "derived_metrics": metadata.get("derived_metrics", {}),
                    "contextual_data": metadata.get("contextual_data", {}),
                    "companion_files": metadata.get("companion_files", {})
                }),
                
                # Processing info
                extraction_method="ProductionMetadataExtractor",
                metadata_completeness_score=self._calculate_completeness_score(metadata),
                extraction_confidence=95.0,  # High confidence for our extractor
                llm_ready=True  # Always set to True for all files
            )
            
            # Save to database
            db.add(file_metadata)
            await db.commit()
            await db.refresh(file_metadata)
            
            return True
            
        except Exception as e:
            print(f"Error extracting and saving metadata: {e}")
            # Log the specific error for debugging
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            await db.rollback()
            return False
    
    def _parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Parse datetime string from EXIF data"""
        if not datetime_str:
            return None
        try:
            return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
        except:
            return None
    
    def _extract_bit_depth_number(self, bit_depth_str: str) -> Optional[int]:
        """Extract numeric bit depth from string like '8-bit RGB'"""
        if not bit_depth_str:
            return None
        try:
            import re
            match = re.search(r'(\d+)-bit', bit_depth_str)
            return int(match.group(1)) if match else None
        except:
            return None
    
    def _determine_equipment_category(self, model: str) -> Optional[str]:
        """Determine equipment category from model name"""
        if not model:
            return None
        
        model_upper = model.upper()
        if any(drone in model_upper for drone in ['DJI', 'PHANTOM', 'MAVIC', 'INSPIRE', 'MATRICE']):
            return "drone"
        elif any(camera in model_upper for camera in ['CANON', 'NIKON', 'SONY', 'LEICA']):
            return "dslr"
        else:
            return "other"
    
    def _calculate_completeness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate metadata completeness score (0-100)"""
        score = 0.0
        total_categories = 10
        
        # Check each category for data
        if metadata.get("gps_data", {}).get("has_gps"):
            score += 15  # GPS is very important
        if metadata.get("camera_settings", {}):
            score += 15  # Camera settings important
        if metadata.get("flight_parameters", {}):
            score += 10  # Flight info important
        if metadata.get("basic_info", {}):
            score += 10  # Basic info always present
        if metadata.get("radiometric_data", {}):
            score += 10  # Color/quality info
        if metadata.get("geometric_data", {}):
            score += 10  # Spatial info
        if metadata.get("elevation_data", {}):
            score += 10  # Elevation info
        if metadata.get("derived_metrics", {}):
            score += 10  # Calculated metrics
        if metadata.get("contextual_data", {}):
            score += 5   # Context info
        if metadata.get("companion_files", {}):
            score += 5   # Additional files
        
        return min(score, 100.0)
    
    async def extract_metadata_async(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata asynchronously
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dict containing all metadata categories
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return self._error_response(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in self.supported_formats:
                return self._error_response(f"Unsupported format: {file_path.suffix}")
            
            # Initialize metadata structure
            metadata = {
                "extraction_info": self._get_extraction_info(),
                "basic_info": {},
                "gps_data": {},
                "geometric_data": {},
                "elevation_data": {},
                "radiometric_data": {},
                "flight_parameters": {},
                "camera_settings": {},
                "quality_assessment": {},
                "derived_metrics": {},
                "contextual_data": {},
                "raw_exif": {},
                "companion_files": {},
                "processing_status": "success"
            }
            
            # Load image and extract EXIF (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            exif_data, img_info = await loop.run_in_executor(
                None, self._extract_image_data, file_path
            )
            
            # Extract each category
            metadata["basic_info"] = await self._extract_basic_info_async(file_path, img_info)
            metadata["gps_data"] = self._extract_gps_data(exif_data)
            metadata["geometric_data"] = await self._extract_geometric_data_async(file_path, exif_data, metadata["basic_info"])
            metadata["elevation_data"] = await self._extract_elevation_data_async(file_path, exif_data)
            metadata["radiometric_data"] = self._extract_radiometric_data(exif_data, {"mode": img_info['mode'], "bands": img_info['bands'], "filename": file_path.name})
            metadata["flight_parameters"] = self._extract_flight_parameters(exif_data)
            metadata["camera_settings"] = self._extract_camera_settings(exif_data)
            metadata["quality_assessment"] = await self._extract_quality_assessment_async(file_path, exif_data)
            metadata["derived_metrics"] = self._calculate_derived_metrics(metadata)
            metadata["contextual_data"] = self._extract_contextual_data(metadata)
            metadata["raw_exif"] = self._clean_exif_data(exif_data)
            metadata["companion_files"] = await self._find_companion_files_async(file_path)
            
            return metadata
            
        except Exception as e:
            return self._error_response(f"Async extraction failed: {str(e)}")
    
    def _extract_image_data(self, file_path: Path) -> tuple:
        """Extract image data and EXIF in executor (blocking operation)"""
        with Image.open(file_path) as img:
            exif_data = self._extract_exif_data(img)
            img_info = {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'bands': img.getbands() if hasattr(img, 'getbands') else None
            }
            return exif_data, img_info
        
    # Legacy sync method - use extract_metadata_async instead
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata and return structured JSON
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dict containing all metadata categories
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return self._error_response(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in self.supported_formats:
                return self._error_response(f"Unsupported format: {file_path.suffix}")
            
            # Initialize metadata structure
            metadata = {
                "extraction_info": self._get_extraction_info(),
                "basic_info": {},
                "gps_data": {},
                "geometric_data": {},
                "elevation_data": {},
                "radiometric_data": {},
                "flight_parameters": {},
                "camera_settings": {},
                "quality_assessment": {},
                "derived_metrics": {},
                "contextual_data": {},
                "raw_exif": {},
                "companion_files": {},
                "processing_status": "success"
            }
            
            # Load image and extract EXIF
            with Image.open(file_path) as img:
                exif_data = self._extract_exif_data(img)
                
                # Extract each category
                metadata["basic_info"] = self._extract_basic_info(file_path, img)
                metadata["gps_data"] = self._extract_gps_data(exif_data)
                metadata["geometric_data"] = self._extract_geometric_data(file_path, exif_data, metadata["basic_info"])
                metadata["elevation_data"] = self._extract_elevation_data(file_path, exif_data)
                metadata["radiometric_data"] = self._extract_radiometric_data(exif_data, {"mode": img.mode, "bands": img.getbands(), "filename": file_path.name})
                metadata["flight_parameters"] = self._extract_flight_parameters(exif_data)
                metadata["camera_settings"] = self._extract_camera_settings(exif_data)
                metadata["quality_assessment"] = self._extract_quality_assessment(file_path, exif_data)
                metadata["derived_metrics"] = self._calculate_derived_metrics(metadata)
                metadata["contextual_data"] = self._extract_contextual_data(metadata)
                metadata["raw_exif"] = self._clean_exif_data(exif_data)
                metadata["companion_files"] = self._find_companion_files(file_path)
                
            return metadata
            
        except Exception as e:
            return self._error_response(f"Extraction failed: {str(e)}")
    
    def _get_extraction_info(self) -> Dict[str, Any]:
        """Get extraction timestamp and version info"""
        return {
            "timestamp": datetime.now().isoformat(),
            "extractor_version": "1.0.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    
    def _extract_exif_data(self, img: Image.Image) -> Dict[str, Any]:
        """Extract raw EXIF data from image"""
        exif_data = {}
        if hasattr(img, '_getexif') and img._getexif() is not None:
            exif = img._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        return exif_data
    
    async def _extract_basic_info_async(self, file_path: Path, img_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic file and image information (async)"""
        loop = asyncio.get_event_loop()
        file_stats = await loop.run_in_executor(None, os.stat, file_path)
        
        return {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": file_stats.st_size,
            "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "file_size_gb": round(file_stats.st_size / (1024 * 1024 * 1024), 4),
            "creation_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modification_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "image_format": img_info['format'],
            "image_mode": img_info['mode'],
            "width_pixels": img_info['size'][0],
            "height_pixels": img_info['size'][1],
            "total_pixels": img_info['size'][0] * img_info['size'][1],
            "megapixels": round((img_info['size'][0] * img_info['size'][1]) / 1000000, 2),
            "aspect_ratio": round(img_info['size'][0] / img_info['size'][1], 4),
            "color_bands": len(img_info['bands']) if img_info['bands'] else None,
            "has_transparency": img_info['mode'] in ['RGBA', 'LA']
        }
    

    
    def _extract_gps_data(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract GPS and location data"""
        gps_data = {
            "has_gps": False,
            "latitude": None,
            "longitude": None,
            "altitude_meters": None,
            "altitude_feet": None,
            "coordinates_decimal": None,
            "coordinates_dms": None,
            "gps_timestamp": None,
            "gps_datestamp": None,
            "gps_status": None,
            "gps_measure_mode": None,
            "speed_kmh": None,
            "speed_mph": None,
            "track_direction": None,
            "image_direction": None,
            "gps_satellites": None,
            "gps_dop": None,
            "altitude_reference": None
        }
        
        if 'GPSInfo' not in exif_data:
            return gps_data
        
        gps_info = exif_data['GPSInfo']
        gps_data["has_gps"] = True
        
        # Extract coordinates
        if 2 in gps_info and 4 in gps_info:  # Latitude and Longitude
            lat = self._convert_gps_coordinate(gps_info[2], gps_info.get(1, 'N'))
            lon = self._convert_gps_coordinate(gps_info[4], gps_info.get(3, 'E'))
            
            if lat is not None and lon is not None:
                gps_data.update({
                    "latitude": lat,
                    "longitude": lon,
                    "coordinates_decimal": f"{lat:.8f}, {lon:.8f}",
                    "coordinates_dms": self._decimal_to_dms(lat, lon)
                })
        
        # Extract altitude
        if 6 in gps_info:
            try:
                altitude_m = float(gps_info[6])
                gps_data.update({
                    "altitude_meters": round(altitude_m, 2),
                    "altitude_feet": round(altitude_m * 3.28084, 2),
                    "altitude_reference": gps_info.get(5, 'Unknown')
                })
            except (ValueError, TypeError):
                # If altitude conversion fails, skip it
                pass
        
        # Extract other GPS fields
        gps_mappings = {
            7: "gps_timestamp",
            9: "gps_status", 
            10: "gps_measure_mode",
            13: "speed_kmh",
            15: "track_direction",
            17: "image_direction",
            8: "gps_satellites",
            11: "gps_dop"
        }
        
        for key, field in gps_mappings.items():
            if key in gps_info:
                value = gps_info[key]
                if field == "speed_kmh" and value:
                    try:
                        gps_data[field] = float(value)
                        gps_data["speed_mph"] = round(float(value) * 0.621371, 2)
                    except (ValueError, TypeError):
                        # If speed conversion fails, skip it
                        pass
                else:
                    gps_data[field] = value
        
        return gps_data
    
    async def _extract_geometric_data_async(self, file_path: Path, exif_data: Dict[str, Any], basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geometric and spatial reference data (async)"""
        geometric = {
            "has_georeferencing": False,
            "coordinate_reference_system": None,
            "projection": None,
            "datum": None,
            "utm_zone": None,
            "epsg_code": None,
            "world_file": {},
            "ground_sample_distance": {},
            "spatial_extents": {},
            "transformation_matrix": None
        }
        
        # Check for world file
        world_file_data = await self._read_world_file_async(file_path)
        if world_file_data:
            geometric["has_georeferencing"] = True
            geometric["world_file"] = world_file_data
            
            # Calculate spatial extents
            if all(key in world_file_data for key in ['pixel_size_x', 'pixel_size_y', 'top_left_x', 'top_left_y']):
                extents = self._calculate_spatial_extents(world_file_data, basic_info)
                geometric["spatial_extents"] = extents
        
        # Check for projection file
        projection_data = await self._read_projection_file_async(file_path)
        if projection_data:
            geometric.update(projection_data)
        
        # Calculate Ground Sample Distance
        gsd_data = self._calculate_gsd(exif_data, basic_info, world_file_data)
        geometric["ground_sample_distance"] = gsd_data
        
        return geometric
    

    
    async def _extract_elevation_data_async(self, file_path: Path, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract elevation and terrain data (async)"""
        elevation = {
            "gps_altitude_meters": None,
            "gps_altitude_feet": None,
            "terrain_classification": None,
            "elevation_files": [],
            "relative_altitude": None,
            "altitude_accuracy": None
        }
        
        # Get GPS altitude
        if 'GPSInfo' in exif_data and 6 in exif_data['GPSInfo']:
            try:
                altitude_m = float(exif_data['GPSInfo'][6])
                elevation.update({
                    "gps_altitude_meters": round(altitude_m, 2),
                    "gps_altitude_feet": round(altitude_m * 3.28084, 2),
                    "terrain_classification": self._classify_terrain(altitude_m)
                })
            except (ValueError, TypeError):
                # If altitude conversion fails, skip it
                pass
        
        # Find companion elevation files
        elevation_files = await self._find_elevation_files_async(file_path)
        elevation["elevation_files"] = elevation_files
        
        return elevation
    

    
    def _extract_radiometric_data(self, exif_data: Dict[str, Any], img_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract radiometric and spectral data"""
        radiometric = {
            "color_space": img_info['mode'],
            "bit_depth": self._get_bit_depth_from_mode(img_info['mode']),
            "spectral_bands": len(img_info['bands']) if img_info['bands'] else None,
            "spectral_type": self._determine_spectral_type(exif_data, img_info.get("filename", "")),
            "color_profile": None,
            "white_balance": None,
            "color_temperature": None,
            "saturation": None,
            "contrast": None,
            "sharpness": None,
            "brightness": None,
            "exposure_bias": None,
            "histogram_stats": {}
        }
        
        # Extract color correction data
        color_fields = {
            'WhiteBalance': 'white_balance',
            'ColorSpace': 'color_profile',
            'ColorTemperature': 'color_temperature', 
            'Saturation': 'saturation',
            'Contrast': 'contrast',
            'Sharpness': 'sharpness',
            'BrightnessValue': 'brightness',
            'ExposureBiasValue': 'exposure_bias'
        }
        
        for exif_key, field_name in color_fields.items():
            if exif_key in exif_data:
                radiometric[field_name] = exif_data[exif_key]
        
        # Note: Histogram statistics would require loading the image again
        # For now, we'll skip this to maintain async performance
        radiometric["histogram_stats"] = {"available": False, "reason": "Skipped for async performance"}
        
        return radiometric
    
    def _extract_flight_parameters(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract drone flight parameters"""
        flight = {
            "drone_manufacturer": None,
            "drone_model": None,
            "drone_serial": None,
            "firmware_version": None,
            "software_version": None,
            "capture_datetime": None,
            "capture_date": None,
            "capture_time": None,
            "flight_id": None,
            "pilot_id": None,
            "mission_name": None,
            "flight_mode": None,
            "gimbal_pitch": None,
            "gimbal_yaw": None,
            "gimbal_roll": None
        }
        
        # Basic drone info
        mapping = {
            'Make': 'drone_manufacturer',
            'Model': 'drone_model', 
            'BodySerialNumber': 'drone_serial',
            'Software': 'software_version',
            'DateTime': 'capture_datetime',
            'DateTimeOriginal': 'capture_datetime'
        }
        
        for exif_key, field_name in mapping.items():
            if exif_key in exif_data:
                flight[field_name] = exif_data[exif_key]
        
        # Parse datetime
        if flight["capture_datetime"]:
            try:
                dt = datetime.strptime(flight["capture_datetime"], "%Y:%m:%d %H:%M:%S")
                flight["capture_date"] = dt.date().isoformat()
                flight["capture_time"] = dt.time().isoformat()
            except:
                pass
        
        return flight
    
    def _extract_camera_settings(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera settings and parameters"""
        camera = {
            "focal_length_mm": None,
            "focal_length_35mm": None,
            "aperture_f_number": None,
            "exposure_time_seconds": None,
            "exposure_time_fraction": None,
            "iso_speed": None,
            "lens_model": None,
            "lens_serial": None,
            "exposure_mode": None,
            "exposure_program": None,
            "metering_mode": None,
            "flash_mode": None,
            "flash_fired": None,
            "digital_zoom_ratio": None,
            "scene_capture_type": None,
            "gain_control": None,
            "subject_distance": None,
            "hyperfocal_distance": None
        }
        
        # Camera settings mapping
        settings_mapping = {
            'FocalLength': 'focal_length_mm',
            'FocalLengthIn35mmFilm': 'focal_length_35mm',
            'FNumber': 'aperture_f_number',
            'ExposureTime': 'exposure_time_fraction',
            'ISOSpeedRatings': 'iso_speed',
            'LensModel': 'lens_model',
            'LensSerialNumber': 'lens_serial',
            'ExposureMode': 'exposure_mode',
            'ExposureProgram': 'exposure_program',
            'MeteringMode': 'metering_mode',
            'Flash': 'flash_mode',
            'DigitalZoomRatio': 'digital_zoom_ratio',
            'SceneCaptureType': 'scene_capture_type',
            'GainControl': 'gain_control',
            'SubjectDistance': 'subject_distance'
        }
        
        for exif_key, field_name in settings_mapping.items():
            if exif_key in exif_data:
                value = exif_data[exif_key]
                
                # Handle different field types appropriately
                if field_name == 'exposure_time_fraction':
                    if isinstance(value, tuple):
                        camera['exposure_time_seconds'] = round(value[0] / value[1], 6)
                        camera[field_name] = f"{value[0]}/{value[1]}"
                    elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                        # Handle IFDRational objects
                        camera['exposure_time_seconds'] = round(value.numerator / value.denominator, 6)
                        camera[field_name] = f"{value.numerator}/{value.denominator}"
                    else:
                        # Convert to string for any other numeric value
                        camera[field_name] = str(value)
                
                elif field_name in ['focal_length_mm', 'focal_length_35mm', 'aperture_f_number']:
                    # Convert to float for numeric fields
                    if value is not None:
                        try:
                            if isinstance(value, tuple):
                                camera[field_name] = float(value[0] / value[1])
                            elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                                camera[field_name] = float(value.numerator / value.denominator)
                            else:
                                camera[field_name] = float(value)
                        except (ValueError, TypeError, ZeroDivisionError):
                            camera[field_name] = None
                    else:
                        camera[field_name] = None
                
                elif field_name == 'iso_speed':
                    # Convert to integer for ISO
                    if value is not None:
                        try:
                            if isinstance(value, (list, tuple)) and len(value) > 0:
                                camera[field_name] = int(value[0])
                            else:
                                camera[field_name] = int(value)
                        except (ValueError, TypeError):
                            camera[field_name] = None
                    else:
                        camera[field_name] = None
                
                else:
                    # Convert other values to strings
                    if value is not None:
                        camera[field_name] = str(value)
                    else:
                        camera[field_name] = None
        
        return camera
    
    async def _extract_quality_assessment_async(self, file_path: Path, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality assessment data (async)"""
        quality = {
            "image_quality_score": None,
            "sharpness_score": None,
            "noise_level": None,
            "compression_quality": None,
            "motion_blur": None,
            "exposure_quality": None,
            "color_accuracy": None,
            "geometric_accuracy": None,
            "companion_reports": []
        }
        
        # Find quality report files
        quality_files = await self._find_quality_files_async(file_path)
        quality["companion_reports"] = quality_files
        
        # Extract quality indicators from EXIF
        if 'Sharpness' in exif_data:
            quality["sharpness_score"] = exif_data['Sharpness']
        
        return quality
    

    
    def _calculate_derived_metrics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived metrics from extracted data"""
        derived = {
            "coverage_area": {},
            "resolution_metrics": {},
            "accuracy_estimates": {},
            "processing_recommendations": []
        }
        
        basic = metadata["basic_info"]
        geometric = metadata["geometric_data"]
        
        # Resolution metrics
        derived["resolution_metrics"] = {
            "pixel_density": f"{basic['width_pixels']}x{basic['height_pixels']}",
            "megapixels": basic["megapixels"],
            "aspect_ratio": basic["aspect_ratio"]
        }
        
        # Coverage area calculations
        if geometric.get("ground_sample_distance", {}).get("average_cm_per_pixel"):
            gsd_cm = geometric["ground_sample_distance"]["average_cm_per_pixel"]
            gsd_m = gsd_cm / 100
            
            area_sqm = (basic["width_pixels"] * gsd_m) * (basic["height_pixels"] * gsd_m)
            
            derived["coverage_area"] = {
                "square_meters": round(area_sqm, 2),
                "square_kilometers": round(area_sqm / 1000000, 6),
                "hectares": round(area_sqm / 10000, 4),
                "acres": round(area_sqm * 0.000247105, 4)
            }
        
        # Processing recommendations
        recommendations = []
        if basic["file_size_mb"] > 100:
            recommendations.append("Consider image compression for faster processing")
        if basic["megapixels"] > 50:
            recommendations.append("High resolution - suitable for detailed analysis")
        if metadata["gps_data"]["has_gps"]:
            recommendations.append("GPS data available - enables georeferenced analysis")
        
        derived["processing_recommendations"] = recommendations
        
        return derived
    
    def _extract_contextual_data(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual information for analysis"""
        contextual = {
            "location_context": {},
            "temporal_context": {},
            "environmental_context": {},
            "equipment_context": {}
        }
        
        # Location context
        if metadata["gps_data"]["has_gps"]:
            lat = metadata["gps_data"]["latitude"]
            lon = metadata["gps_data"]["longitude"]
            altitude = metadata["gps_data"]["altitude_meters"]
            
            contextual["location_context"] = {
                "coordinates": f"{lat:.6f}, {lon:.6f}",
                "region": self._determine_region(lat, lon),
                "altitude_meters": altitude,
                "climate_zone": self._determine_climate(lat, lon),
                "terrain_type": metadata["elevation_data"]["terrain_classification"]
            }
        
        # Temporal context
        flight_params = metadata["flight_parameters"]
        if flight_params["capture_date"]:
            contextual["temporal_context"] = {
                "capture_date": flight_params["capture_date"],
                "capture_time": flight_params["capture_time"],
                "season": self._determine_season(flight_params["capture_date"]),
                "time_of_day": self._classify_time_of_day(flight_params["capture_time"])
            }
        
        # Equipment context
        basic_info = metadata["basic_info"]
        radiometric_data = metadata["radiometric_data"]
        
        # Check thermal capability from both drone model and file characteristics
        is_thermal_from_drone = self._is_thermal_drone(flight_params["drone_model"])
        is_thermal_from_file = self._is_thermal_file(basic_info.get("filename", ""), radiometric_data.get("spectral_type", ""))
        is_thermal_capable = is_thermal_from_drone or is_thermal_from_file
        
        contextual["equipment_context"] = {
            "device_model": flight_params["drone_model"] or flight_params["drone_manufacturer"],
            "is_thermal_capable": is_thermal_capable,
            "is_professional_grade": self._is_professional_drone(flight_params["drone_model"]),
            "recommended_for_analysis": True
        }
        
        return contextual
    
    async def _find_companion_files_async(self, file_path: Path) -> Dict[str, List[str]]:
        """Find companion files related to the image (async)"""
        companions = {
            "world_files": [],
            "projection_files": [],
            "elevation_files": [],
            "quality_reports": [],
            "metadata_files": [],
            "calibration_files": []
        }
        
        base_name = file_path.stem
        folder = file_path.parent
        
        # Define file patterns
        patterns = {
            "world_files": ['.tfw', '.jgw', '.pgw', '.bpw'],
            "projection_files": ['.prj'],
            "elevation_files": ['_dem.tif', '_dsm.tif', '_dtm.tif', '_elevation.tif'],
            "quality_reports": ['_quality.xml', '_report.pdf', '_processing.log'],
            "metadata_files": ['.xml', '.txt', '.met'],
            "calibration_files": ['_calibration.txt', '_accuracy.txt']
        }
        
        # Run file existence checks in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        for category, extensions in patterns.items():
            for ext in extensions:
                if ext.startswith('_'):
                    # Suffix pattern
                    potential_file = folder / (base_name + ext)
                else:
                    # Extension pattern
                    potential_file = file_path.with_suffix(ext)
                
                # Check existence asynchronously
                exists = await loop.run_in_executor(None, potential_file.exists)
                if exists:
                    companions[category].append(str(potential_file))
        
        return companions
    
    async def _find_elevation_files_async(self, file_path: Path) -> List[str]:
        """Find companion elevation files (async)"""
        elevation_files = []
        base_name = file_path.stem
        folder = file_path.parent
        
        elevation_patterns = [
            '_dem.tif', '_dsm.tif', '_dtm.tif', '_elevation.tif',
            '_heights.tif', '_dem.tiff', '_dsm.tiff', '_dtm.tiff'
        ]
        
        loop = asyncio.get_event_loop()
        
        for pattern in elevation_patterns:
            potential_file = folder / (base_name + pattern)
            exists = await loop.run_in_executor(None, potential_file.exists)
            if exists:
                elevation_files.append(str(potential_file))
        
        return elevation_files
    
    async def _find_quality_files_async(self, file_path: Path) -> List[str]:
        """Find quality assessment files (async)"""
        quality_files = []
        base_name = file_path.stem
        folder = file_path.parent
        
        quality_patterns = [
            '_quality.xml', '_report.pdf', '_processing.log',
            '_calibration.txt', '_accuracy.txt', '_metadata.xml'
        ]
        
        loop = asyncio.get_event_loop()
        
        for pattern in quality_patterns:
            potential_file = folder / (base_name + pattern)
            exists = await loop.run_in_executor(None, potential_file.exists)
            if exists:
                quality_files.append(str(potential_file))
        
        return quality_files
    

    
    # Helper methods for data conversion and calculations
    
    def _convert_gps_coordinate(self, coord_tuple: tuple, ref: str) -> Optional[float]:
        """Convert GPS coordinates from DMS to decimal degrees"""
        if not coord_tuple or len(coord_tuple) != 3:
            return None
        
        try:
            degrees, minutes, seconds = coord_tuple
            decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
            
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return round(decimal, 8)
        except:
            return None
    
    def _decimal_to_dms(self, lat: float, lon: float) -> str:
        """Convert decimal degrees to DMS format"""
        def dd_to_dms(dd):
            degrees = int(dd)
            minutes = int((dd - degrees) * 60)
            seconds = ((dd - degrees) * 60 - minutes) * 60
            return f"{abs(degrees)}°{minutes}'{seconds:.2f}\""
        
        lat_dms = dd_to_dms(lat)
        lon_dms = dd_to_dms(lon)
        lat_ref = 'N' if lat >= 0 else 'S'
        lon_ref = 'E' if lon >= 0 else 'W'
        
        return f"{lat_dms}{lat_ref}, {lon_dms}{lon_ref}"
    
    async def _read_world_file_async(self, file_path: Path) -> Dict[str, Any]:
        """Read world file for georeferencing (async)"""
        world_ext = self.world_file_extensions.get(file_path.suffix.lower())
        
        if not world_ext:
            return {}
        
        world_file = file_path.with_suffix(world_ext)
        if not world_file.exists():
            return {}
        
        try:
            async with aiofiles.open(world_file, 'r') as f:
                content = await f.read()
                lines = [float(line.strip()) for line in content.splitlines()[:6]]
            
            if len(lines) >= 6:
                return {
                    "file_path": str(world_file),
                    "pixel_size_x": lines[0],
                    "rotation_x": lines[1],
                    "rotation_y": lines[2], 
                    "pixel_size_y": abs(lines[3]),
                    "top_left_x": lines[4],
                    "top_left_y": lines[5],
                    "units": "meters"  # Assumption
                }
        except Exception:
            pass
        
        return {}
    

    
    async def _read_projection_file_async(self, file_path: Path) -> Dict[str, Any]:
        """Read projection file for CRS information (async)"""
        prj_file = file_path.with_suffix('.prj')
        
        if not prj_file.exists():
            return {}
        
        try:
            async with aiofiles.open(prj_file, 'r') as f:
                wkt_string = (await f.read()).strip()
            
            projection_info = {
                "projection_file_path": str(prj_file),
                "wkt_string": wkt_string
            }
            
            # Parse basic info from WKT
            if 'UTM' in wkt_string.upper():
                import re
                utm_match = re.search(r'UTM.*Zone.*(\d+)', wkt_string)
                if utm_match:
                    projection_info["utm_zone"] = int(utm_match.group(1))
                    projection_info["coordinate_reference_system"] = f"UTM Zone {utm_match.group(1)}"
            
            if 'WGS84' in wkt_string or 'WGS_1984' in wkt_string:
                projection_info["datum"] = "WGS84"
            
            return projection_info
            
        except Exception:
            return {}
    

    
    def _calculate_gsd(self, exif_data: Dict[str, Any], basic_info: Dict[str, Any], world_file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Ground Sample Distance"""
        gsd = {
            "method": None,
            "x_cm_per_pixel": None,
            "y_cm_per_pixel": None,
            "average_cm_per_pixel": None,
            "accuracy": None
        }
        
        # Method 1: From world file
        if world_file_data and 'pixel_size_x' in world_file_data:
            gsd_x = abs(world_file_data['pixel_size_x']) * 100  # Convert to cm
            gsd_y = abs(world_file_data['pixel_size_y']) * 100
            
            gsd.update({
                "method": "world_file",
                "x_cm_per_pixel": round(gsd_x, 4),
                "y_cm_per_pixel": round(gsd_y, 4),
                "average_cm_per_pixel": round((gsd_x + gsd_y) / 2, 4),
                "accuracy": "high"
            })
            
        # Method 2: From EXIF resolution
        elif 'XResolution' in exif_data and 'YResolution' in exif_data:
            x_res = exif_data['XResolution']
            y_res = exif_data['YResolution']
            res_unit = exif_data.get('ResolutionUnit', 2)
            
            if isinstance(x_res, (int, float)) and x_res > 0:
                if res_unit == 2:  # inches
                    gsd_x = 2.54 / x_res
                    gsd_y = 2.54 / y_res
                elif res_unit == 3:  # cm
                    gsd_x = 1.0 / x_res
                    gsd_y = 1.0 / y_res
                else:
                    gsd_x = 2.54 / x_res
                    gsd_y = 2.54 / y_res
                
                gsd.update({
                    "method": "exif_resolution",
                    "x_cm_per_pixel": round(gsd_x, 4),
                    "y_cm_per_pixel": round(gsd_y, 4),
                    "average_cm_per_pixel": round((gsd_x + gsd_y) / 2, 4),
                    "accuracy": "medium"
                })
        
        return gsd
    
    def _calculate_spatial_extents(self, world_file_data: Dict[str, Any], basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate spatial extents from world file data"""
        width = basic_info['width_pixels']
        height = basic_info['height_pixels']
        
        top_left_x = world_file_data['top_left_x']
        top_left_y = world_file_data['top_left_y']
        pixel_size_x = world_file_data['pixel_size_x']
        pixel_size_y = world_file_data['pixel_size_y']
        
        return {
            "top_left": [top_left_x, top_left_y],
            "top_right": [top_left_x + (width * pixel_size_x), top_left_y],
            "bottom_left": [top_left_x, top_left_y - (height * pixel_size_y)],
            "bottom_right": [top_left_x + (width * pixel_size_x), top_left_y - (height * pixel_size_y)],
            "center": [top_left_x + (width * pixel_size_x / 2), top_left_y - (height * pixel_size_y / 2)],
            "min_x": top_left_x,
            "max_x": top_left_x + (width * pixel_size_x),
            "min_y": top_left_y - (height * pixel_size_y),
            "max_y": top_left_y,
            "width_meters": abs(width * pixel_size_x),
            "height_meters": abs(height * pixel_size_y),
            "area_square_meters": abs(width * pixel_size_x * height * pixel_size_y)
        }
    
    def _find_elevation_files(self, file_path: Path) -> List[str]:
        """Find companion elevation files"""
        elevation_files = []
        base_name = file_path.stem
        folder = file_path.parent
        
        elevation_patterns = [
            '_dem.tif', '_dsm.tif', '_dtm.tif', '_elevation.tif',
            '_heights.tif', '_dem.tiff', '_dsm.tiff', '_dtm.tiff'
        ]
        
        for pattern in elevation_patterns:
            potential_file = folder / (base_name + pattern)
            if potential_file.exists():
                elevation_files.append(str(potential_file))
        
        return elevation_files
    
    def _find_quality_files(self, file_path: Path) -> List[str]:
        """Find quality assessment files"""
        quality_files = []
        base_name = file_path.stem
        folder = file_path.parent
        
        quality_patterns = [
            '_quality.xml', '_report.pdf', '_processing.log',
            '_calibration.txt', '_accuracy.txt', '_metadata.xml'
        ]
        
        for pattern in quality_patterns:
            potential_file = folder / (base_name + pattern)
            if potential_file.exists():
                quality_files.append(str(potential_file))
        
        return quality_files
    
    def _get_bit_depth_from_mode(self, mode: str) -> str:
        """Determine bit depth from image mode"""
        mode_mapping = {
            'L': '8-bit grayscale',
            'P': '8-bit palette',
            'RGB': '8-bit RGB',
            'RGBA': '8-bit RGBA',
            'CMYK': '8-bit CMYK',
            'YCbCr': '8-bit YCbCr',
            'LAB': '8-bit LAB',
            'HSV': '8-bit HSV',
            'I': '32-bit integer',
            'F': '32-bit float'
        }
        return mode_mapping.get(mode, f"{mode} (unknown)")
    
    def _get_bit_depth(self, img: Image.Image) -> str:
        """Determine bit depth of image"""
        mode_mapping = {
            'L': '8-bit grayscale',
            'P': '8-bit palette',
            'RGB': '8-bit RGB',
            'RGBA': '8-bit RGBA',
            'CMYK': '8-bit CMYK',
            'YCbCr': '8-bit YCbCr',
            'LAB': '8-bit LAB',
            'HSV': '8-bit HSV',
            'I': '32-bit integer',
            'F': '32-bit float'
        }
        return mode_mapping.get(img.mode, f"{img.mode} (unknown)")
    
    def _determine_spectral_type(self, exif_data: Dict[str, Any], filename: str = None) -> str:
        """Determine spectral type from EXIF data and filename"""
        model = exif_data.get('Model', '').upper()
        
        # Check EXIF model first
        if 'THERMAL' in model or 'IR' in model or 'M3T' in model or 'FLIR' in model:
            return 'thermal_infrared'
        elif 'MULTISPECTRAL' in model or 'NIR' in model:
            return 'multispectral'
        elif 'LIDAR' in model:
            return 'lidar'
        
        # If no model in EXIF, check filename
        if filename:
            filename_upper = filename.upper()
            if any(indicator in filename_upper for indicator in ['THERMAL', 'IR', 'FLIR', 'RADIOMETRIC', 'TEMP', 'HEAT']):
                return 'thermal_infrared'
            elif any(indicator in filename_upper for indicator in ['MULTISPECTRAL', 'NIR', 'NDVI']):
                return 'multispectral'
            elif any(indicator in filename_upper for indicator in ['LIDAR', 'POINTCLOUD']):
                return 'lidar'
        
        return 'rgb_visible'
    
    def _calculate_histogram_stats(self, img: Image.Image) -> Dict[str, Any]:
        """Calculate histogram statistics"""
        try:
            if img.mode == 'RGB':
                r, g, b = img.split()
                
                # Calculate basic statistics
                r_hist = r.histogram()
                g_hist = g.histogram()
                b_hist = b.histogram()
                
                return {
                    "available": True,
                    "red_channel_peak": r_hist.index(max(r_hist)),
                    "green_channel_peak": g_hist.index(max(g_hist)),
                    "blue_channel_peak": b_hist.index(max(b_hist)),
                    "red_channel_mean": sum(i * v for i, v in enumerate(r_hist)) / sum(r_hist),
                    "green_channel_mean": sum(i * v for i, v in enumerate(g_hist)) / sum(g_hist),
                    "blue_channel_mean": sum(i * v for i, v in enumerate(b_hist)) / sum(b_hist),
                    "dynamic_range": "0-255 (8-bit)"
                }
            elif img.mode == 'L':
                hist = img.histogram()
                return {
                    "available": True,
                    "grayscale_peak": hist.index(max(hist)),
                    "grayscale_mean": sum(i * v for i, v in enumerate(hist)) / sum(hist),
                    "dynamic_range": "0-255 (8-bit grayscale)"
                }
        except:
            pass
        
        return {"available": False}
    
    def _classify_terrain(self, altitude_m: float) -> str:
        """Classify terrain based on altitude"""
        if altitude_m < 100:
            return "coastal_lowland"
        elif altitude_m < 300:
            return "plains_low_hills"
        elif altitude_m < 800:
            return "hills_plateaus"
        elif altitude_m < 1500:
            return "mountainous"
        else:
            return "high_mountain"
    
    def _determine_region(self, lat: float, lon: float) -> str:
        """Determine geographic region from coordinates"""
        # Simple region classification for India
        if 8 <= lat <= 37 and 68 <= lon <= 97:
            if 20 <= lat <= 30 and 70 <= lon <= 88:
                return "central_india"
            elif 12 <= lat <= 20 and 74 <= lon <= 80:
                return "southern_india"
            elif 8 <= lat <= 12 and 76 <= lon <= 80:
                return "far_southern_india"
            else:
                return "india"
        else:
            return "unknown_region"
    
    def _determine_climate(self, lat: float, lon: float) -> str:
        """Determine climate zone"""
        if 8 <= lat <= 37 and 68 <= lon <= 97:
            if lat > 30:
                return "temperate"
            elif lat > 20:
                return "subtropical"
            else:
                return "tropical"
        return "unknown_climate"
    
    def _determine_season(self, date_str: str) -> str:
        """Determine season from date (India-specific)"""
        try:
            date_obj = datetime.fromisoformat(date_str)
            month = date_obj.month
            
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "summer"
            elif month in [6, 7, 8, 9]:
                return "monsoon"
            else:
                return "post_monsoon"
        except:
            return "unknown_season"
    
    def _classify_time_of_day(self, time_str: str) -> str:
        """Classify time of day"""
        try:
            time_obj = datetime.fromisoformat(f"2000-01-01T{time_str}").time()
            hour = time_obj.hour
            
            if 5 <= hour < 12:
                return "morning"
            elif 12 <= hour < 17:
                return "afternoon"
            elif 17 <= hour < 20:
                return "evening"
            else:
                return "night"
        except:
            return "unknown_time"
    
    def _is_thermal_drone(self, model: str) -> bool:
        """Check if drone model has thermal capability"""
        if not model:
            return False
        
        thermal_indicators = ['M3T', 'THERMAL', 'IR', 'FLIR', 'RADIOMETRIC']
        return any(indicator in model.upper() for indicator in thermal_indicators)
    
    def _is_thermal_file(self, filename: str, spectral_type: str) -> bool:
        """Check if file is thermal based on filename and spectral type"""
        if not filename:
            return False
        
        filename_upper = filename.upper()
        thermal_filename_indicators = ['THERMAL', 'IR', 'FLIR', 'RADIOMETRIC', 'TEMP', 'HEAT']
        
        # Check filename for thermal indicators
        if any(indicator in filename_upper for indicator in thermal_filename_indicators):
            return True
        
        # Check spectral type
        if spectral_type and spectral_type.lower() in ['thermal_infrared', 'ir', 'thermal']:
            return True
        
        return False
    
    def _is_professional_drone(self, model: str) -> bool:
        """Check if drone/camera is professional grade"""
        if not model:
            return False
        
        pro_indicators = [
            'PHANTOM 4 PRO', 'MAVIC 3', 'M3T', 'INSPIRE', 'MATRICE', 'M300', 'M600',
            'CANON EOS', 'NIKON D', 'SONY A7', 'PHASE ONE', 'HASSELBLAD', 'LEICA'
        ]
        return any(indicator in model.upper() for indicator in pro_indicators)
    
    def _clean_exif_data(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean EXIF data for JSON serialization"""
        cleaned = {}
        for key, value in exif_data.items():
            cleaned[key] = self._clean_value_for_json(value)
        return cleaned
    
    def _clean_value_for_json(self, value: Any) -> Any:
        """Recursively clean values for JSON serialization"""
        if value is None:
            return None
        elif isinstance(value, (int, float, bool)):
            return value
        elif isinstance(value, str):
            # Remove null bytes and other problematic characters
            cleaned = value.replace('\x00', '').replace('\u0000', '')
            # Remove other control characters that might cause issues
            cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
            return cleaned
        elif isinstance(value, (list, tuple)):
            return [self._clean_value_for_json(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._clean_value_for_json(v) for k, v in value.items()}
        elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            # Handle IFDRational objects
            return f"{value.numerator}/{value.denominator}"
        elif hasattr(value, '__str__'):
            # Convert to string and clean
            str_value = str(value)
            return self._clean_value_for_json(str_value)
        else:
            return str(value)
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Return standardized error response"""
        return {
            "extraction_info": self._get_extraction_info(),
            "processing_status": "error",
            "error_message": error_message,
            "basic_info": {},
            "gps_data": {"has_gps": False},
            "geometric_data": {"has_georeferencing": False},
            "elevation_data": {},
            "radiometric_data": {},
            "flight_parameters": {},
            "camera_settings": {},
            "quality_assessment": {},
            "derived_metrics": {},
            "contextual_data": {},
            "raw_exif": {},
            "companion_files": {}
        }
    
    def save_metadata_json(self, metadata: Dict[str, Any], output_path: str) -> bool:
        """Save metadata to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            print("Attempting to clean data for JSON output...")
            try:
                cleaned_metadata = self._clean_value_for_json(metadata)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_metadata, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e2:
                print(f"Error saving cleaned JSON: {e2}")
                return False
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def format_summary_report(self, metadata: Dict[str, Any]) -> str:
        """Format a human-readable summary report"""
        if metadata["processing_status"] == "error":
            return f"❌ Error: {metadata['error_message']}"
        
        basic = metadata["basic_info"]
        gps = metadata["gps_data"]
        geometric = metadata["geometric_data"]
        derived = metadata["derived_metrics"]
        
        report = f"""
🗺️ IMAGE METADATA SUMMARY
{'='*50}

📁 FILE INFORMATION:
   • File: {basic['filename']}
   • Size: {basic['file_size_mb']} MB
   • Dimensions: {basic['width_pixels']}x{basic['height_pixels']} ({basic['megapixels']} MP)
   • Format: {basic['image_format']} ({basic['image_mode']})

📍 LOCATION DATA:
   • GPS Available: {'✅ Yes' if gps['has_gps'] else '❌ No'}"""

        if gps['has_gps']:
            report += f"""
   • Coordinates: {gps['coordinates_decimal']}
   • Altitude: {gps['altitude_meters']}m ({gps['altitude_feet']}ft)"""
        
        if geometric['has_georeferencing']:
            report += f"""

📐 GEOREFERENCING:
   • World File: {'✅ Found' if geometric['world_file'] else '❌ Not found'}
   • CRS: {geometric.get('coordinate_reference_system', 'Not specified')}"""
            
            if geometric.get('ground_sample_distance', {}).get('average_cm_per_pixel'):
                gsd = geometric['ground_sample_distance']['average_cm_per_pixel']
                report += f"""
   • Ground Sample Distance: {gsd} cm/pixel"""
        
        if derived.get('coverage_area', {}).get('square_meters'):
            area = derived['coverage_area']
            report += f"""

📏 COVERAGE AREA:
   • Area: {area['square_meters']} m² ({area['acres']} acres)"""
        
        flight = metadata["flight_parameters"]
        if flight.get('drone_model'):
            report += f"""

🚁 EQUIPMENT:
   • Device: {flight.get('drone_manufacturer', '')} {flight['drone_model']}
   • Capture: {flight.get('capture_date', '')} {flight.get('capture_time', '')}"""
        
        context = metadata["contextual_data"]
        if context.get('location_context', {}).get('region'):
            report += f"""

🌍 CONTEXT:
   • Region: {context['location_context']['region']}
   • Climate: {context['location_context'].get('climate_zone', 'Unknown')}
   • Season: {context.get('temporal_context', {}).get('season', 'Unknown')}"""
        
        report += f"""

✅ Processing Status: {metadata['processing_status'].upper()}
📊 Extracted at: {metadata['extraction_info']['timestamp']}
"""
        
        return report

