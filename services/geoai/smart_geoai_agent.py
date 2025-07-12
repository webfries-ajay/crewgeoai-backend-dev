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
        """Analyze multiple selected images with comprehensive comparison"""
        try:
            if not self.selected_file_paths:
                yield "❌ No images selected for analysis."
                return
                
            print(f"🖼️ Starting multi-image analysis for {len(self.selected_file_paths)} images")
            
            # For streaming, we'll analyze images sequentially and stream each result
            yield f"🔍 MULTI-IMAGE ANALYSIS ({len(self.selected_file_paths)} images)\n\n"
            
            for i, (file_id, file_info) in enumerate(self.selected_file_paths.items(), 1):
                yield f"📸 **Image {i}/{len(self.selected_file_paths)}: {file_info['filename']}**\n"
                # (REMOVED) Metadata context output
                # Only show image name and count
                enhanced_question = f"""Multi-image analysis ({i}/{len(self.selected_file_paths)}): {file_info['filename']}\n\n{question}\n\nPlease provide detailed analysis for this specific image, keeping in mind this is part of a multi-image comparison."""
                async for chunk in self.analyze_image(file_info['path'], enhanced_question):
                    yield chunk
                if i < len(self.selected_file_paths):
                    yield "\n\n---\n\n"
            yield f"\n\n🎯 **MULTI-IMAGE ANALYSIS COMPLETE**\nAnalyzed {len(self.selected_file_paths)} images successfully!"
        except Exception as e:
            print(f"❌ Error in multi-image analysis: {str(e)}")
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
            
            # Choose between single or multi-image analysis
            if has_selected_files and len(self.selected_file_paths) > 1:
                print(f"🔍 Multi-image analysis for {len(self.selected_file_paths)} images")
                async for chunk in self.analyze_multiple_images(question):
                    yield chunk
            elif has_current_image:
                print(f"🔍 Single image analysis: {self.current_image}")
                async for chunk in self.analyze_image(self.current_image, question):
                    yield chunk
            elif has_selected_files:
                # Single selected file
                file_info = list(self.selected_file_paths.values())[0]
                print(f"🔍 Single selected file analysis: {file_info['filename']}")
                async for chunk in self.analyze_image(file_info['path'], question):
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