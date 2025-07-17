import os
import sys
import re
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from services.geoai.unified_langchain_analyzer import UnifiedLangChainAnalyzer
from typing import AsyncIterator

# Load config from .example
EXAMPLE_PATH = Path(__file__).parent / ".example"
if EXAMPLE_PATH.exists():
    with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip()

class SmartGeoAIAgent:
    def __init__(self, master_prompt: str = None, project_id: str = None, user_id: str = None):
        """Initialize SmartGeoAIAgent with category-specific prompt and conversation context"""
        
        # Use unified LangChain analyzer with category-specific prompt and conversation context
        self.analyzer = UnifiedLangChainAnalyzer(
            master_prompt=master_prompt,
            project_id=project_id,
            user_id=user_id
        )
        
        self.current_image = None
        self.selected_file_ids = []  # Support for multiple selected files
        self.selected_file_paths = {}  # Dictionary of file_id -> {path, filename}
        self.selected_file_metadata = {}  # Dictionary of file_id -> metadata dict
        self.project_id = project_id
        self.user_id = user_id
        self.master_prompt = master_prompt or "You are an intelligent GeoAI specialist capable of analyzing any type of imagery."
        
        print(f"🤖 SmartGeoAIAgent initialized")
        print(f"🧠 Using {'category-specific' if master_prompt else 'default'} master prompt")
        print(f"📚 Conversation context: {'enabled' if project_id and user_id else 'disabled'}")
        print(f"🔗 Ready for image analysis and chat with streaming support")
    
    def extract_image_path(self, text):
        """Extract image path from text"""
        match = re.search(r'(\S+\.(?:jpg|jpeg|png|tif|tiff))', text, re.IGNORECASE)
        if match and os.path.exists(match.group(1)):
            return match.group(1)
        return None
    
    def _format_metadata_context(self, metadata: dict, filename: str) -> str:
        """Format metadata into readable context for LLM"""
        if not metadata:
            print(f"📋 No metadata provided for {filename}")
            return ""
        
        print(f"📋 Formatting metadata for {filename}: {metadata}")
        context_parts = []
        
        # Basic file info
        context_parts.append(f"📁 Filename: {filename}")
        
        # Camera information
        if metadata.get('camera_make') or metadata.get('camera_model'):
            camera_info = []
            if metadata.get('camera_make'):
                camera_info.append(metadata['camera_make'])
            if metadata.get('camera_model'):
                camera_info.append(metadata['camera_model'])
            context_parts.append(f"📷 Camera: {' '.join(camera_info)}")
        
        # Technical settings
        tech_settings = []
        if metadata.get('focal_length'):
            tech_settings.append(f"Focal Length: {metadata['focal_length']}mm")
        if metadata.get('aperture'):
            tech_settings.append(f"Aperture: f/{metadata['aperture']}")
        if metadata.get('iso_speed'):
            tech_settings.append(f"ISO: {metadata['iso_speed']}")
        if tech_settings:
            context_parts.append(f"⚙️ Settings: {', '.join(tech_settings)}")
        
        # GPS information
        if metadata.get('has_gps') and metadata.get('latitude') and metadata.get('longitude'):
            context_parts.append(f"📍 Location: {metadata['latitude']:.6f}, {metadata['longitude']:.6f}")
            if metadata.get('altitude'):
                context_parts.append(f"🏔️ Altitude: {metadata['altitude']}m")
        
        # Date information
        if metadata.get('date_taken'):
            context_parts.append(f"📅 Date: {metadata['date_taken']}")
        
        # Image quality
        if metadata.get('image_quality'):
            context_parts.append(f"🎨 Quality: {metadata['image_quality']}")
        if metadata.get('bit_depth'):
            context_parts.append(f"🔢 Bit Depth: {metadata['bit_depth']}-bit")
        
        # Equipment classification
        if metadata.get('equipment_category'):
            context_parts.append(f"🚁 Equipment: {metadata['equipment_category']}")
        if metadata.get('is_professional_grade'):
            context_parts.append(f"⭐ Professional Grade: {'Yes' if metadata['is_professional_grade'] else 'No'}")
        
        # Metadata quality
        if metadata.get('metadata_completeness_score'):
            context_parts.append(f"📊 Metadata Completeness: {metadata['metadata_completeness_score']:.1f}%")
        if metadata.get('extraction_confidence'):
            context_parts.append(f"🎯 Extraction Confidence: {metadata['extraction_confidence']:.1f}%")
        
        # Processed metadata (if available)
        if metadata.get('processed_metadata'):
            processed = metadata['processed_metadata']
            if isinstance(processed, dict):
                # Basic info
                if processed.get('basic_info'):
                    basic = processed['basic_info']
                    if basic.get('megapixels'):
                        context_parts.append(f"📐 Resolution: {basic['megapixels']} MP")
                    if basic.get('aspect_ratio'):
                        context_parts.append(f"📏 Aspect Ratio: {basic['aspect_ratio']}")
                
                # Spectral type
                if processed.get('radiometric_data', {}).get('spectral_type'):
                    context_parts.append(f"🌈 Spectral Type: {processed['radiometric_data']['spectral_type']}")
                
                # Equipment context
                if processed.get('contextual_data', {}).get('equipment_context', {}).get('is_thermal_capable'):
                    context_parts.append("🔥 Thermal Capable: Yes")
        
        formatted_context = "\n".join(context_parts)
        print(f"📋 Final formatted context: {formatted_context}")
        return formatted_context
    
    def _get_color_mapping_description(self, index_type: str) -> str:
        """Generate detailed color mapping description for NDVI/NDMI interpretation"""
        if index_type.lower() == 'ndvi':
            return """
🎨 **NDVI COLOR MAPPING GUIDE:**
   • **Brown (RGB: 139,69,19)**: NDVI 0.0-0.2 - Very poor/sparse vegetation, bare soil, dead vegetation
   • **Orange (RGB: 255,165,0)**: NDVI 0.2-0.3 - Poor vegetation, stressed crops, minimal chlorophyll
   • **Yellow (RGB: 255,255,0)**: NDVI 0.3-0.5 - Moderate vegetation, developing crops, moderate chlorophyll
   • **Yellow-Green (RGB: 173,255,47)**: NDVI 0.5-0.7 - Good vegetation, healthy crops, good chlorophyll activity
   • **Green (RGB: 0,255,0)**: NDVI 0.7-0.9 - Very good vegetation, vigorous crops, high chlorophyll
   • **Dark Green (RGB: 0,100,0)**: NDVI 0.9-1.0 - Excellent vegetation, maximum biomass, peak chlorophyll
   
📖 **INTERPRETATION GUIDE:** Darker greens = healthier/denser vegetation, Browns/oranges = poor/stressed vegetation"""
        
        elif index_type.lower() == 'ndmi':
            return """
🎨 **NDMI COLOR MAPPING GUIDE:**
   • **Brown (RGB: 139,69,19)**: NDMI 0.0-0.2 - Very dry conditions, severe drought stress, low moisture
   • **Dark Orange (RGB: 255,140,0)**: NDMI 0.2-0.3 - Dry conditions, drought stress, below optimal moisture
   • **Gold (RGB: 255,215,0)**: NDMI 0.3-0.5 - Moderate moisture, transitional conditions
   • **Yellow (RGB: 255,255,0)**: NDMI 0.5-0.7 - Good moisture, adequate water content
   • **Light Blue (RGB: 173,216,230)**: NDMI 0.7-0.9 - High moisture, well-hydrated vegetation
   • **Deep Blue (RGB: 0,191,255)**: NDMI 0.9-1.0 - Very high moisture, optimal hydration, possible waterlogging
   
📖 **INTERPRETATION GUIDE:** Darker blues = higher moisture/better hydration, Browns/oranges = dry/drought conditions"""
        
        return ""

    async def analyze_image(self, image_path: str, question: str) -> AsyncIterator[str]:
        """Analyze single image using intelligent detection and comprehensive analysis"""
        try:
            print(f"🔍 Starting intelligent image analysis for: {image_path}")
            print(f"📝 Question: {question}")
            
            # Validate file
            if not self.analyzer.validate_file(image_path):
                yield f"❌ Error: Image file '{image_path}' not found or invalid."
                return
            
            # Get file info
            file_info = self.analyzer.get_file_info(image_path)
            print(f"📊 File info: {file_info['format']}, {file_info['size']}, {file_info['file_size_mb']:.2f}MB")
            
            # Find metadata for this image
            metadata_context = ""
            for file_id, file_path_info in self.selected_file_paths.items():
                if file_path_info['path'] == image_path:
                    if file_id in self.selected_file_metadata and self.selected_file_metadata[file_id]:
                        metadata = self.selected_file_metadata[file_id]
                        print(f"📋 Raw metadata received: {metadata}")
                        metadata_context = self._format_metadata_context(metadata, file_path_info['filename'])
                        print(f"📋 Formatted metadata context: {metadata_context}")
                        print(f"📋 Found metadata for: {file_path_info['filename']}")
                    else:
                        print(f"📋 No metadata found for file_id: {file_id}")
                    break
            
            # Enhance question with metadata context if available
            enhanced_question = question
            if metadata_context:
                enhanced_question = f"""Image Analysis Request: {question}

📋 IMAGE METADATA CONTEXT:
{metadata_context}

Please use this metadata context to provide more accurate and detailed analysis of the image."""
                print(f"🔍 Enhanced question with metadata context")
            
            # Print the enhanced question being sent to LLM
            print(f"📤 ENHANCED QUESTION SENT TO LLM:")
            print(f"   Original: {question}")
            print(f"   Enhanced: {enhanced_question}")
            print(f"   Metadata Context Length: {len(metadata_context) if metadata_context else 0} characters")
            
            print(f"🧠 Using intelligent detection and analysis system")
            
            # Use the unified LangChain analyzer with intelligent detection and streaming
            async for chunk in self.analyzer.analyze_image_stream(image_path, enhanced_question):
                yield chunk
            
            print(f"✅ Intelligent analysis completed")
                
        except Exception as e:
            print(f"❌ Error in analyze_image: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"❌ Error analyzing image: {str(e)}"
    
    async def _get_processed_image_statistics(self, file_id: str, index_type: str) -> str:
        """Retrieve processed image statistics from database and format for LLM"""
        try:
            from sqlalchemy.ext.asyncio import AsyncSession
            from sqlalchemy import select
            from models.processed_image import ProcessedImage, ProcessedImageType
            from core.database import get_db
            
            # Get database session (this is a simplified approach - in production you'd pass this)
            # For now, we'll try to get it from the analyzer if available
            if hasattr(self.analyzer, 'db') and self.analyzer.db:
                db = self.analyzer.db
            else:
                # Fallback: return placeholder text if no DB access
                return f"Statistical data for {index_type.upper()} is being processed..."
            
            # Query for the processed image
            result = await db.execute(
                select(ProcessedImage).where(
                    ProcessedImage.file_id == file_id,
                    ProcessedImage.processed_image_type == ProcessedImageType(index_type.lower())
                )
            )
            processed_image = result.scalar_one_or_none()
            
            if not processed_image or not processed_image.processing_stats:
                return f"No statistical data available for {index_type.upper()}"
            
            # Extract LLM summary from processing stats
            llm_summary = processed_image.processing_stats.get("summary_for_llm", {})
            statistics = processed_image.processing_stats.get("statistics", {})
            
            if llm_summary.get("error"):
                return f"{index_type.upper()} Statistical Error: {llm_summary['error']}"
            
            # Format statistics for LLM consumption
            stats_text = f"""
📈 {index_type.upper()} QUANTITATIVE ANALYSIS:

Overall Assessment: {llm_summary.get('overall_health', 'Unknown').replace('_', ' ').title()}

Key Metrics:
• Mean Value: {llm_summary.get('key_metrics', {}).get('mean_value', 0):.3f}
• Value Range: {llm_summary.get('key_metrics', {}).get('range', 'Unknown')}
• Standard Deviation: {llm_summary.get('key_metrics', {}).get('std_deviation', 0):.3f}
• Valid Coverage: {llm_summary.get('key_metrics', {}).get('valid_coverage', 'Unknown')}
"""
            
            # Add index-specific interpretations
            interpretation = llm_summary.get('interpretation', {})
            
            if index_type == "ndvi":
                vegetation_coverage = interpretation.get('vegetation_coverage', {})
                land_use = interpretation.get('land_use_breakdown', {})
                
                stats_text += f"""
Vegetation Health Distribution:
• Healthy Vegetation: {vegetation_coverage.get('healthy_vegetation', 'Unknown')}
• Stressed Vegetation: {vegetation_coverage.get('stressed_vegetation', 'Unknown')}
• Health Score: {vegetation_coverage.get('health_score', 0):.3f}

Land Use Classification:
"""
                for class_name, percentage in land_use.items():
                    formatted_name = class_name.replace('_', ' ').title()
                    stats_text += f"• {formatted_name}: {percentage}\n"
                    
            elif index_type == "ndmi":
                moisture_status = interpretation.get('moisture_status', {})
                moisture_dist = interpretation.get('moisture_distribution', {})
                
                stats_text += f"""
Moisture Status Distribution:
• Well Hydrated Areas: {moisture_status.get('well_hydrated', 'Unknown')}
• Drought Stressed Areas: {moisture_status.get('drought_stress', 'Unknown')}
• Average Moisture Score: {moisture_status.get('moisture_score', 0):.3f}

Moisture Classification:
"""
                for class_name, percentage in moisture_dist.items():
                    formatted_name = class_name.replace('_', ' ').title()
                    stats_text += f"• {formatted_name}: {percentage}\n"
            
            # Add spatial analysis if available
            spatial_patterns = llm_summary.get('spatial_patterns', {})
            if spatial_patterns:
                stats_text += f"""
Spatial Distribution:
• Significant Patches: {spatial_patterns.get('total_patches', 0)}
• Average Patch Size: {spatial_patterns.get('average_patch_size', 'Unknown')}
• Spatial Uniformity: {spatial_patterns.get('spatial_uniformity', 'Unknown').title()}
"""
            
            # Add percentile data for advanced analysis
            percentiles = statistics.get('percentiles', {})
            if percentiles:
                stats_text += f"""
Statistical Percentiles (for threshold analysis):
• 25th percentile: {percentiles.get('p25', 0):.3f}
• 75th percentile: {percentiles.get('p75', 0):.3f}
• 90th percentile: {percentiles.get('p90', 0):.3f}
"""
            
            return stats_text.strip()
            
        except Exception as e:
            print(f"[STATS] Error retrieving {index_type} statistics: {e}")
            return f"Error retrieving {index_type.upper()} statistical data: {str(e)}"
    
    async def chat_with_llm(self, message: str) -> AsyncIterator[str]:
        """Handle general chat using unified LangChain analyzer with intelligent context"""
        try:
            print(f"💬 Using unified LangChain chat with intelligent context")
            print(f"📤 TEXT-ONLY CHAT DATA SENT TO LLM: {message}")
            
            # Use the unified analyzer's streaming chat method
            async for chunk in self.analyzer.chat_with_context_stream(message):
                yield chunk
            
        except Exception as e:
            yield f"❌ Error in chat: {str(e)}"

    async def analyze_multiple_images(self, question: str) -> AsyncIterator[str]:
        """Analyze multiple selected images with comprehensive comparison including NDVI/NDMI"""
        try:
            if not self.selected_file_paths:
                yield "❌ No images selected for analysis."
                return
                
            # Group images by parent file ID to organize original, NDVI, and NDMI together
            grouped_images = {}
            for file_key, file_info in self.selected_file_paths.items():
                parent_id = file_info.get('parent_file_id', file_key)
                if parent_id not in grouped_images:
                    grouped_images[parent_id] = {}
                
                image_type = file_info.get('image_type', 'original')
                grouped_images[parent_id][image_type] = {
                    'key': file_key,
                    'info': file_info
                }
            
            print(f"🖼️ Starting enhanced multi-image analysis for {len(grouped_images)} file groups")
            print(f"📊 Image breakdown: {sum(len(group) for group in grouped_images.values())} total images")
            
            # For streaming, we'll analyze image groups and provide combined analysis
            yield f"🔍 **COMPREHENSIVE MULTI-IMAGE ANALYSIS**\n"
            yield f"📊 **Analyzing {len(grouped_images)} file groups with enhanced agricultural intelligence**\n\n"
            
            for group_idx, (parent_id, image_group) in enumerate(grouped_images.items(), 1):
                # Get the original filename for group header
                original_info = image_group.get('original', {}).get('info', {})
                group_name = original_info.get('filename', f"File Group {group_idx}").replace(" (Original)", "")
                
                yield f"📸 **ANALYSIS {group_idx}/{len(grouped_images)}: {group_name}**\n"
                
                # Show available image types in this group
                available_types = []
                if 'original' in image_group:
                    available_types.append("Original")
                if 'ndvi' in image_group:
                    available_types.append("NDVI")
                if 'ndmi' in image_group:
                    available_types.append("NDMI")
                
                yield f"🔍 **Processing imagery**: {', '.join(available_types)}\n\n"
                
                # Collect analysis results from all image types
                analysis_results = {}
                image_paths_for_combined = []
                
                # Process images quietly to collect data
                for image_type in ['original', 'ndvi', 'ndmi']:
                    if image_type in image_group:
                        image_data = image_group[image_type]
                        file_info = image_data['info']
                        image_paths_for_combined.append({
                            'path': file_info['path'],
                            'type': image_type,
                            'filename': file_info['filename']
                        })
                        
                        print(f"[COMBINED ANALYSIS] Processing {image_type} image: {file_info['filename']}")
                        
                        # Create specific analysis question for data collection
                        if image_type == 'original':
                            analysis_question = f"""Analyze this original agricultural image focusing on:
1. Visible crop conditions and health
2. Field structure and layout
3. Any equipment or infrastructure
4. Potential defects or issues
5. Object detection for all visible elements

Provide detailed technical analysis for data collection purposes."""
                        elif image_type == 'ndvi':
                            color_guide = self._get_color_mapping_description('ndvi')
                            analysis_question = f"""Analyze this NDVI (vegetation index) image with professional color mapping:

{color_guide}

**Analysis Focus:**
1. Vegetation health patterns using the specific color meanings above
2. Biomass distribution across different color zones
3. Areas of concern or stress (browns/oranges indicate problems)
4. Growth uniformity and spatial patterns
5. Agricultural insights based on precise color-to-value relationships

**Important:** Reference specific colors and their corresponding NDVI values/meanings in your analysis. Use the RGB values and NDVI ranges provided above to give precise interpretations."""
                        elif image_type == 'ndmi':
                            color_guide = self._get_color_mapping_description('ndmi')
                            analysis_question = f"""Analyze this NDMI (moisture index) image with professional color mapping:

{color_guide}

**Analysis Focus:**
1. Moisture content patterns using the specific color meanings above
2. Irrigation effectiveness across different color zones
3. Water stress indicators (browns/oranges indicate drought)
4. Soil moisture distribution and spatial patterns
5. Drought or waterlogging areas based on color zones

**Important:** Reference specific colors and their corresponding NDMI values/meanings in your analysis. Use the RGB values and NDMI ranges provided above to give precise moisture interpretations."""
                        
                        # Collect analysis result silently
                        individual_result = ""
                        async for chunk in self.analyze_image(file_info['path'], analysis_question):
                            individual_result += chunk
                        
                        analysis_results[image_type] = individual_result
                        print(f"[COMBINED ANALYSIS] Collected {len(individual_result)} characters from {image_type}")
                
                # Now create comprehensive combined analysis
                yield "🧠 **Generating comprehensive analysis...**\n\n"
                
                # Extract key insights and statistics without full individual analyses
                statistical_data = {}
                color_guides = {}
                defect_data = []
                
                # Process each image type to extract essential data only
                if 'ndvi' in analysis_results:
                    ndvi_stats_text = await self._get_processed_image_statistics(image_group['ndvi']['info']['parent_file_id'], 'ndvi')
                    ndvi_color_guide = self._get_color_mapping_description('ndvi')
                    statistical_data['ndvi'] = ndvi_stats_text
                    color_guides['ndvi'] = ndvi_color_guide
                    
                    # Extract defects from NDVI analysis
                    ndvi_analysis = analysis_results['ndvi']
                    defect_start = ndvi_analysis.find("---DEFECT_DATA_START---")
                    defect_end = ndvi_analysis.find("---DEFECT_DATA_END---")
                    if defect_start != -1 and defect_end != -1:
                        defect_section = ndvi_analysis[defect_start:defect_end + len("---DEFECT_DATA_END---")]
                        defect_data.append(defect_section)
                
                if 'ndmi' in analysis_results:
                    ndmi_stats_text = await self._get_processed_image_statistics(image_group['ndmi']['info']['parent_file_id'], 'ndmi')
                    ndmi_color_guide = self._get_color_mapping_description('ndmi')
                    statistical_data['ndmi'] = ndmi_stats_text
                    color_guides['ndmi'] = ndmi_color_guide
                    
                    # Extract defects from NDMI analysis
                    ndmi_analysis = analysis_results['ndmi']
                    defect_start = ndmi_analysis.find("---DEFECT_DATA_START---")
                    defect_end = ndmi_analysis.find("---DEFECT_DATA_END---")
                    if defect_start != -1 and defect_end != -1:
                        defect_section = ndmi_analysis[defect_start:defect_end + len("---DEFECT_DATA_END---")]
                        defect_data.append(defect_section)
                
                if 'original' in analysis_results:
                    # Extract defects from original analysis
                    original_analysis = analysis_results['original']
                    defect_start = original_analysis.find("---DEFECT_DATA_START---")
                    defect_end = original_analysis.find("---DEFECT_DATA_END---")
                    if defect_start != -1 and defect_end != -1:
                        defect_section = original_analysis[defect_start:defect_end + len("---DEFECT_DATA_END---")]
                        defect_data.append(defect_section)
                
                # Create streamlined combined analysis prompt
                combined_prompt = f"""UNIFIED AGRICULTURAL INTELLIGENCE ANALYSIS

USER QUESTION: {question}

CONTEXT: I have analyzed this agricultural image using three different methods:
- Original Visual Analysis: Crop conditions, field structure, visible issues
- NDVI Analysis: Vegetation health and biomass distribution  
- NDMI Analysis: Moisture content and irrigation effectiveness

STATISTICAL DATA AVAILABLE:
"""
                
                if 'ndvi' in statistical_data:
                    combined_prompt += f"""
🌱 NDVI STATISTICS & COLOR MAPPING:
{statistical_data['ndvi']}

{color_guides['ndvi']}
"""
                
                if 'ndmi' in statistical_data:
                    combined_prompt += f"""
💧 NDMI STATISTICS & COLOR MAPPING:
{statistical_data['ndmi']}

{color_guides['ndmi']}
"""
                
                combined_prompt += f"""
DETECTED ISSUES:
{chr(10).join(defect_data) if defect_data else "No critical defects detected in preliminary analysis."}

TASK: Create ONE comprehensive agricultural analysis that synthesizes all available data sources.

REQUIREMENTS:
1. **Single Unified Response** - Do not create separate sections for each image type
2. **Professional Agricultural Assessment** - Focus on practical farming insights
3. **Statistical Integration** - Reference specific percentages and classifications from the data
4. **Color-Based Insights** - Use the color mapping guides to interpret vegetation and moisture patterns
5. **Defect Analysis** - Include any detected issues with proper bounding box format
6. **Actionable Recommendations** - Provide specific steps for crop management

RESPONSE STRUCTURE:
- **Comprehensive Agricultural Analysis**: Combined insights from visual + NDVI + NDMI data
- **Statistical Insights**: Key metrics and classifications with percentages
- **Defect Detection**: Any issues found with locations (maintain ---DEFECT_DATA_START--- format)
- **Recommendations**: Specific agricultural actions based on the complete analysis

Create a professional, data-driven response that provides maximum value to farmers by combining all analysis perspectives into actionable intelligence."""
                


                # Generate combined analysis using the unified analyzer
                async for chunk in self.analyzer.chat_with_context_stream(combined_prompt):
                    yield chunk
                
                if group_idx < len(grouped_images):
                    yield "\n\n" + "="*50 + "\n\n"
            
            # Final summary
            total_images = sum(len(group) for group in grouped_images.values())
            total_ndvi = sum(1 for group in grouped_images.values() if 'ndvi' in group)
            total_ndmi = sum(1 for group in grouped_images.values() if 'ndmi' in group)
            
            yield f"\n\n🎯 **COMPREHENSIVE ANALYSIS COMPLETE**\n"
            yield f"📊 **Data Sources Used:**\n"
            yield f"   • File groups analyzed: {len(grouped_images)}\n"
            yield f"   • Total images processed: {total_images}\n"
            yield f"   • Visual data (Original): {len(grouped_images)}\n"
            yield f"   • Vegetation data (NDVI): {total_ndvi}\n"
            yield f"   • Moisture data (NDMI): {total_ndmi}\n"
            yield f"\n🚀 **Enhanced agricultural intelligence combining visual, vegetation health, and moisture analysis!**"
            
        except Exception as e:
            print(f"❌ Error in comprehensive multi-image analysis: {str(e)}")
            yield f"❌ Error analyzing multiple images: {str(e)}"

    async def process_message(self, user_input: str) -> AsyncIterator[str]:
        """Process user message and determine appropriate action with streaming response"""
        
        question = user_input.strip()
        
        # Check what images are available
        has_current_image = self.current_image is not None
        has_selected_files = len(self.selected_file_paths) > 0
        
        print(f"🤔 Processing message (streaming): '{question[:50]}...'")
        print(f"🖼️ Image context: current={has_current_image}, selected={len(self.selected_file_paths) if has_selected_files else 0}")
        
        # Print the data being sent to LLM
        llm_data = {
            "user_message": question,
            "has_current_image": has_current_image,
            "has_selected_files": has_selected_files,
            "selected_file_count": len(self.selected_file_paths),
            "selected_files": list(self.selected_file_paths.keys()) if self.selected_file_paths else [],
            "file_metadata_available": list(self.selected_file_metadata.keys()) if self.selected_file_metadata else [],
            "project_id": self.project_id,
            "user_id": self.user_id,
            "master_prompt_length": len(self.master_prompt) if self.master_prompt else 0
        }
        print(f"📤 DATA SENT TO LLM: {llm_data}")
        
        # Simple logic: If images are selected, use image mode; otherwise, use chat mode
        if has_current_image or has_selected_files:
            print(f"🔍 Images available - using IMAGE ANALYSIS mode")
            
            # Check if we have multiple image types for enhanced analysis
            if has_selected_files:
                # Group images by parent file ID to check for multiple image types
                grouped_images = {}
                for file_key, file_info in self.selected_file_paths.items():
                    parent_id = file_info.get('parent_file_id', file_key)
                    if parent_id not in grouped_images:
                        grouped_images[parent_id] = {}
                    
                    image_type = file_info.get('image_type', 'original')
                    grouped_images[parent_id][image_type] = {
                        'key': file_key,
                        'info': file_info
                    }
                
                # Check if we have multiple files OR multiple image types for any file
                has_multiple_groups = len(grouped_images) > 1
                has_multiple_types = any(len(group) > 1 for group in grouped_images.values())
                
                if has_multiple_groups or has_multiple_types:
                    print(f"🔍 Enhanced multi-image analysis: {len(grouped_images)} groups, enhanced types: {has_multiple_types}")
                async for chunk in self.analyze_multiple_images(question):
                    yield chunk
                else:
                    # Single file, single image type - use traditional analysis
                    file_info = list(self.selected_file_paths.values())[0]
                    print(f"🔍 Single image analysis: {file_info['filename']}")
                    async for chunk in self.analyze_image(file_info['path'], question):
                        yield chunk
            elif has_current_image:
                print(f"🔍 Current image analysis: {self.current_image}")
                async for chunk in self.analyze_image(self.current_image, question):
                    yield chunk
            else:
                yield "❌ Error: No valid images found for analysis."
                
            # Add conversation to history for image analysis (collect full response)
            if self.project_id and self.user_id:
                # Note: Conversation history is already handled in the analyzer methods
                print("💾 Conversation history managed by analyzer")
        else:
            print(f"💬 No images selected - using TEXT CHAT mode")
            async for chunk in self.chat_with_llm(user_input):
                yield chunk
            # Note: chat_with_llm handles conversation history internally