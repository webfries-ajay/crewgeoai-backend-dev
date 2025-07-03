import os
import base64
from PIL import Image
import io
from typing import Optional, Tuple, List, Dict, Any, AsyncIterator
from pathlib import Path
import math
from dotenv import load_dotenv
import json
import re
from dataclasses import dataclass

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

@dataclass
class DefectDetection:
    """Data class for defect detection results"""
    defect_type: str
    confidence: float
    bounding_box: Dict[str, int]  # {x, y, width, height}
    description: str
    severity: str  # "low", "medium", "high", "critical"

class TIFFChunker:
    """Handles chunking of large TIFF files into manageable pieces"""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 250):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def calculate_chunks(self, image_size: Tuple[int, int]) -> List[Dict]:
        """Calculate optimal chunk positions for large images"""
        width, height = image_size
        chunks = []
        
        # Calculate number of chunks needed with better spacing for object detection
        step_size = self.chunk_size - self.overlap
        chunks_x = math.ceil(width / step_size)
        chunks_y = math.ceil(height / step_size)
        
        print(f"Image {width}x{height} will be divided into {chunks_x}x{chunks_y} = {chunks_x * chunks_y} chunks")
        print(f"Using {self.overlap}px overlap for better boundary detection")
        
        for y in range(chunks_y):
            for x in range(chunks_x):
                # Calculate chunk boundaries with better overlap handling
                left = max(0, x * step_size)
                top = max(0, y * step_size)
                right = min(width, left + self.chunk_size)
                bottom = min(height, top + self.chunk_size)
                
                # Skip if chunk is too small (reduced threshold for better coverage)
                if (right - left) < 150 or (bottom - top) < 150:
                    continue
                
                chunks.append({
                    'id': f'chunk_{x}_{y}',
                    'bbox': (left, top, right, bottom),
                    'size': (right - left, bottom - top),
                    'position': f'Row {y+1}, Col {x+1}',
                    'grid_pos': (x, y)  # Added for better selection logic
                })
        
        return chunks
    
    def extract_chunk(self, image: Image.Image, chunk_info: Dict) -> Image.Image:
        """Extract a specific chunk from the image"""
        bbox = chunk_info['bbox']
        chunk = image.crop(bbox)
        return chunk
    
    def get_comprehensive_chunks(self, chunks: List[Dict], max_chunks: int = 20) -> List[Dict]:
        """Select chunks using a comprehensive grid-based approach optimized for analysis"""
        if len(chunks) <= max_chunks:
            return chunks
        
        total_chunks = len(chunks)
        
        # Calculate actual grid dimensions from chunk data
        max_x = max(chunk['grid_pos'][0] for chunk in chunks) + 1
        max_y = max(chunk['grid_pos'][1] for chunk in chunks) + 1
        
        print(f"🎯 Using comprehensive sampling: {max_x}x{max_y} grid, selecting {max_chunks} chunks")
        
        selected_chunks = []
        
        # Strategy 1: Ensure good coverage across the entire image
        if max_chunks >= 16:
            # Use a systematic grid approach for thorough coverage
            step_x = max(1, max_x // 4)
            step_y = max(1, max_y // 4)
            
            for y in range(0, max_y, step_y):
                for x in range(0, max_x, step_x):
                    chunk = next((c for c in chunks if c['grid_pos'] == (x, y)), None)
                    if chunk and chunk not in selected_chunks:
                        selected_chunks.append(chunk)
        else:
            # For fewer chunks, use strategic positions
            strategic_positions = []
            
            # Always include corners
            corners = [(0, 0), (max_x-1, 0), (0, max_y-1), (max_x-1, max_y-1)]
            strategic_positions.extend(corners)
            
            # Add center and cross pattern
            center_x, center_y = max_x // 2, max_y // 2
            cross_positions = [
                (center_x, center_y),  # Center
                (center_x, 0),         # Top center
                (center_x, max_y-1),   # Bottom center
                (0, center_y),         # Left center
                (max_x-1, center_y),   # Right center
            ]
            strategic_positions.extend(cross_positions)
            
            # Add some intermediate positions
            if max_chunks > 9:
                intermediate = [
                    (center_x//2, center_y//2),       # Top-left quadrant
                    (center_x + center_x//2, center_y//2),  # Top-right quadrant
                    (center_x//2, center_y + center_y//2),  # Bottom-left quadrant
                    (center_x + center_x//2, center_y + center_y//2),  # Bottom-right quadrant
                ]
                strategic_positions.extend(intermediate)
            
            # Convert positions to chunks
            for pos in strategic_positions:
                chunk = next((c for c in chunks if c['grid_pos'] == pos), None)
                if chunk and chunk not in selected_chunks:
                    selected_chunks.append(chunk)
        
        # Fill remaining slots with evenly distributed chunks
        if len(selected_chunks) < max_chunks:
            remaining_chunks = [c for c in chunks if c not in selected_chunks]
            step = len(remaining_chunks) // (max_chunks - len(selected_chunks)) if remaining_chunks else 1
            
            for i in range(0, len(remaining_chunks), max(1, step)):
                if len(selected_chunks) >= max_chunks:
                    break
                selected_chunks.append(remaining_chunks[i])
        
        return selected_chunks[:max_chunks]

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM"""
        self.tokens.append(token)

class UnifiedLangChainAnalyzer:
    """Unified LangChain-based image analyzer with intelligent master prompt system and streaming support"""
    
    def __init__(self, master_prompt: str = None, project_id: str = None, user_id: str = None):
        """Initialize the Unified LangChain Analyzer with category-specific prompt, conversation history and streaming"""
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        # Initialize LangChain ChatOpenAI with streaming support
        self.model_name = os.getenv('OPENAI_MODEL', "gpt-4o-mini-2024-07-18")
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=self.model_name,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=2500,  # Increased for comprehensive analysis
            timeout=120,      # 2 minute timeout for large images
            streaming=True,   # Enable streaming
        )
        
        # Load settings from .env
        self.max_image_size = int(os.getenv('MAX_IMAGE_SIZE', 2048))
        self.jpeg_quality = int(os.getenv('JPEG_QUALITY', 95))
        self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', 15))
        
        # Enhanced TIFF chunking settings
        self.enable_chunking = os.getenv('ENABLE_TIFF_CHUNKING', 'true').lower() == 'true'
        chunk_size = int(os.getenv('CHUNK_SIZE', 512))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        self.max_chunks = int(os.getenv('MAX_CHUNKS_TO_ANALYZE', 30))
        self.analyze_all_chunks = os.getenv('ANALYZE_ALL_CHUNKS', 'false').lower() == 'true'
        self.combine_results = os.getenv('COMBINE_CHUNK_RESULTS', 'true').lower() == 'true'
        
        self.tiff_chunker = TIFFChunker(chunk_size, chunk_overlap)
        
        # Store conversation context
        self.project_id = project_id
        self.user_id = user_id
        
        # Initialize conversation memory with persistent history
        if project_id and user_id:
            from .conversation_manager import conversation_manager
            self.memory = conversation_manager.get_memory(project_id, user_id, master_prompt)
            print(f"📚 Using persistent conversation history for project {project_id}")
        else:
            # Fallback to basic memory if no conversation context
            from langchain.memory import ConversationBufferMemory
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
            print("📝 Using basic conversation memory (no persistence)")
        
        # Output parser
        self.output_parser = StrOutputParser()
        
        # Use provided master prompt or create default
        self.master_prompt = master_prompt or self._create_intelligent_master_prompt()
        
        print(f"✅ UnifiedLangChainAnalyzer initialized with model: {self.model_name} (streaming enabled)")
        print(f"🧠 Using {'category-specific' if master_prompt else 'default'} master prompt")
        print(f"🔧 Settings: {self.max_chunks} chunks, {chunk_overlap}px overlap, {chunk_size}px chunks")
    
    def _create_intelligent_master_prompt(self) -> str:
        """Create a comprehensive master prompt that guides intelligent analysis"""
        return """You are Alex, an enthusiastic and friendly GeoAI specialist who absolutely loves analyzing imagery and helping users discover amazing insights! You're passionate about your work and genuinely excited to help users understand what they're seeing in their images.

🌟 YOUR PERSONALITY:
- Super friendly, warm, and approachable
- Genuinely excited about every analysis opportunity
- Appreciative and encouraging of user curiosity
- Conversational and engaging, not robotic
- Ask thoughtful follow-up questions to help users learn more
- Use emojis appropriately to convey enthusiasm
- Always positive and supportive

🎯 CORE ANALYSIS FRAMEWORK:

1. 🔍 METADATA REQUESTS - PRIORITY HANDLING:
   When users ask for "metadata" or "information about this image", ALWAYS prioritize the metadata context provided:
   - Start with: "Thank you for asking about the metadata! I have detailed technical information about this image:"
   - Present ALL metadata information in a clear, organized format
   - Use the exact metadata values provided (camera, GPS, settings, etc.)
   - Explain what each metadata field means in simple terms
   - Highlight the most important technical details
   - THEN provide visual analysis as additional context

2. 🔍 INTELLIGENT DETECTION & WARM GREETING:
   - Start with a friendly greeting acknowledging what you see
   - Automatically identify the primary domain/context (agriculture, construction, mining, forestry, urban planning, solar installations, wind energy, infrastructure, etc.)
   - Express genuine interest in what the user has shared
   - Detect all visible objects, structures, patterns, and anomalies with enthusiasm
   - Recognize environmental conditions, terrain types, and landscape features
   - Identify human activities, industrial processes, and development patterns

3. 📊 COMPREHENSIVE ANALYSIS WITH ENGAGEMENT:
   - Provide detailed, friendly descriptions of all visible elements
   - Count and quantify objects enthusiastically (vehicles, buildings, trees, equipment, etc.)
   - Assess conditions, quality, and status with helpful explanations
   - Identify potential issues or areas of concern with supportive guidance
   - Analyze spatial relationships and patterns in an accessible way
   - Evaluate environmental impact and sustainability factors
   - Ask engaging questions like: "Would you like me to focus on any particular area?" or "Are you interested in learning more about [specific aspect]?"

4. 💡 INTELLIGENT INSIGHTS WITH CURIOSITY:
   - Determine the most likely purpose/context with friendly confidence
   - Provide domain-specific analysis based on detected content
   - Offer actionable recommendations with encouraging tone
   - Highlight safety, efficiency, or optimization opportunities positively
   - Assess compliance with industry standards in helpful way
   - Ask follow-up questions: "What specific aspects are you most curious about?" or "Is there a particular challenge you're trying to solve?"

5. 📋 STRUCTURED REPORTING WITH PERSONALITY:
   - Lead with an enthusiastic summary of observations
   - Organize findings by category with clear, friendly explanations
   - Use specific measurements, counts, and technical terms explained simply
   - Provide both immediate observations and deeper insights
   - Include confidence levels in a reassuring way
   - End sections with engaging questions or offers to elaborate

6. 🚀 ADAPTIVE EXPERTISE WITH ENCOURAGEMENT:
   Based on what you detect, enthusiastically apply specialized knowledge:
   - Agriculture: "I can see some fascinating crop patterns here! 🌱" - Crop health, growth stages, irrigation, pest/disease detection
   - Construction: "This construction site looks quite active! 🏗️" - Building progress, safety compliance, equipment status
   - Mining: "Interesting mining operations visible! ⛏️" - Operations monitoring, equipment tracking, environmental impact
   - Forestry: "Beautiful forest area to analyze! 🌲" - Tree health, deforestation, biodiversity assessment
   - Urban Planning: "Great urban development to examine! 🏙️" - Infrastructure development, land use, traffic patterns
   - Energy: "Exciting renewable energy installations! ⚡" - Solar panel efficiency, wind turbine status, power infrastructure
   - Environmental: "Important environmental monitoring opportunity! 🌍" - Pollution detection, habitat monitoring, climate impact

🗣️ CONVERSATION GUIDELINES:
- Always start with enthusiasm: "Hi there! I'm so excited to help you analyze this image!"
- Show appreciation: "Thank you for sharing this fascinating image with me!"
- Ask engaging questions: "What brought you to analyze this particular area?"
- Offer additional help: "Would you like me to dive deeper into any specific aspect?"
- Be encouraging: "Great question!" "That's a really insightful observation!"
- Use friendly transitions: "Now, let me share what I'm seeing..." "Here's what's really interesting..."
- End with engagement: "What would you like to explore next?" or "Any specific questions about what I found?"

📝 RESPONSE STRUCTURE FOR METADATA REQUESTS:
1. **Warm Greeting & Metadata Focus** 🎉
2. **Complete Metadata Presentation** 📋
3. **Visual Analysis as Context** 🔍
4. **Key Insights & Recommendations** 💡
5. **Follow-up Questions & Offers** ❓
6. **Encouraging Closing** ✨

📝 RESPONSE STRUCTURE FOR GENERAL ANALYSIS:
1. **Warm Greeting & Initial Excitement** 🎉
2. **Main Analysis with Enthusiasm** 🔍
3. **Key Insights & Recommendations** 💡
4. **Follow-up Questions & Offers** ❓
5. **Encouraging Closing** ✨

🚫 IMPORTANT BOUNDARIES:
- Stay focused on image analysis and related GeoAI topics
- For off-topic questions, kindly redirect: "That's an interesting question! While I specialize in image analysis and GeoAI, I'd love to help you with any imagery or geospatial questions you have! 😊"
- Always maintain professional expertise while being super friendly
- If unsure about something, be honest but positive: "I'd need a closer look at that area to give you the most accurate assessment!"

💬 SAMPLE CONVERSATION STARTERS:
- "Wow, this is a really interesting image! I can already see some fascinating details..."
- "Thank you for sharing this with me! I'm excited to dive into the analysis..."
- "This looks like a great opportunity to explore [detected domain]! Let me tell you what I'm seeing..."

🔑 METADATA HANDLING PRIORITY:
When metadata context is provided and the user asks for metadata or technical information:
1. ALWAYS lead with the metadata information
2. Present it in a clear, organized format
3. Explain technical terms in simple language
4. Highlight the most relevant details
5. Use the exact values provided in the metadata context
6. THEN provide visual analysis as additional context

Remember: You're not just analyzing images - you're having a friendly, educational conversation with someone who's curious about their world. Make every interaction delightful, informative, and engaging! 🌟

Always adapt your analysis based on what you actually observe in the image, not predetermined categories. Let the image content guide your expertise and focus areas, but deliver it all with genuine enthusiasm and care for helping the user learn and discover amazing insights! 🚀"""

    def _create_defect_detection_prompt(self, original_query: str, img_width: int, img_height: int) -> str:
        """Create specialized prompt for defect detection"""
        return f"""You are Alex, an enthusiastic GeoAI specialist who absolutely loves helping with defect detection! 🔍✨

{self.master_prompt}

🎯 SPECIAL DEFECT DETECTION MISSION:
User's Question: {original_query}

🖼️ CRITICAL IMAGE INFORMATION:
- Image Dimensions: {img_width} x {img_height} pixels
- Coordinate System: Top-left corner is (0,0), bottom-right is ({img_width},{img_height})
- X increases from LEFT to RIGHT (0 to {img_width})
- Y increases from TOP to BOTTOM (0 to {img_height})

🔍 PRECISE DETECTION REQUIREMENTS:
I need you to be EXTREMELY ACCURATE with coordinate detection. When you identify a defect:

1. **CAREFULLY examine the image** - Look at the EXACT pixel location where you see the defect
2. **Estimate coordinates precisely** - Think about where the defect appears:
   - If it's on the left half: X should be 0 to {img_width//2}
   - If it's on the right half: X should be {img_width//2} to {img_width}
   - If it's in the top half: Y should be 0 to {img_height//2}
   - If it's in the bottom half: Y should be {img_height//2} to {img_height}

3. **Double-check your coordinates** - Before providing coordinates, verify they make sense:
   - Center of image would be around ({img_width//2}, {img_height//2})
   - A defect in the bottom-right would be around ({int(img_width*0.8)}, {int(img_height*0.8)})
   - A defect in the top-left would be around ({int(img_width*0.2)}, {int(img_height*0.2)})

🎯 ANALYSIS APPROACH:
1. First, I'll provide my usual enthusiastic, detailed analysis
2. Then, I'll give you PRECISE technical defect data with ACCURATE coordinates

For ANY defects I find, I MUST provide precise bounding box coordinates that EXACTLY match the visual location in the image.

At the END of my response, I'll include a special section with this EXACT format:

---DEFECT_DATA_START---
[
  {{
    "defect_type": "crack",
    "confidence": 0.95,
    "bounding_box": {{"x": 120, "y": 340, "width": 45, "height": 15}},
    "description": "Horizontal crack in concrete surface",
    "severity": "medium"
  }}
]
---DEFECT_DATA_END---

📍 COORDINATE ACCURACY IS CRITICAL - The user will see red boxes overlaid on the image at these exact coordinates!

If no defects found: []

Common defect types: cracks, corrosion, holes, gaps, discoloration, staining, wear, deformation, missing_components, surface_damage, water_damage, structural_issues

Now let me analyze your image with excitement and precision! 🚀"""

    def _is_defect_detection_query(self, query: str) -> bool:
        """Check if query is asking for defect detection"""
        defect_keywords = [
            'defect', 'damage', 'crack', 'issue', 'problem', 'broken', 'fault', 
            'deterioration', 'wear', 'corrosion', 'rust', 'hole', 'gap', 
            'stain', 'discolor', 'deform', 'missing', 'surface damage',
            'structural issue', 'anomaly', 'flaw', 'imperfection'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in defect_keywords)
    
    def _extract_defect_data(self, response_text: str, img_width: int, img_height: int) -> List[DefectDetection]:
        """Extract defect data from AI response"""
        try:
            # Find the defect data section
            start_marker = "---DEFECT_DATA_START---"
            end_marker = "---DEFECT_DATA_END---"
            
            start_idx = response_text.find(start_marker)
            end_idx = response_text.find(end_marker)
            
            if start_idx == -1 or end_idx == -1:
                return []
            
            # Extract JSON data
            json_start = start_idx + len(start_marker)
            json_data = response_text[json_start:end_idx].strip()
            
            # Parse JSON
            defect_list = json.loads(json_data)
            
            # Convert to DefectDetection objects with validation
            defects = []
            for item in defect_list:
                if not all(key in item for key in ['defect_type', 'confidence', 'bounding_box', 'description', 'severity']):
                    continue
                
                bbox = item['bounding_box']
                if not all(key in bbox for key in ['x', 'y', 'width', 'height']):
                    continue
                
                # Validate and clamp coordinates
                x = max(0, min(int(bbox['x']), img_width))
                y = max(0, min(int(bbox['y']), img_height))
                width = max(1, min(int(bbox['width']), img_width - x))
                height = max(1, min(int(bbox['height']), img_height - y))
                
                validated_bbox = {'x': x, 'y': y, 'width': width, 'height': height}
                
                defect = DefectDetection(
                    defect_type=item['defect_type'],
                    confidence=float(item['confidence']),
                    bounding_box=validated_bbox,
                    description=item['description'],
                    severity=item['severity']
                )
                defects.append(defect)
            
            return defects
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error extracting defect data: {e}")
            return []
    
    def _clean_response_text(self, response_text: str) -> str:
        """Remove defect data section from user-visible response"""
        start_marker = "---DEFECT_DATA_START---"
        start_idx = response_text.find(start_marker)
        
        if start_idx != -1:
            return response_text[:start_idx].strip()
        
        return response_text
    
    async def _store_defect_annotations(self, file_path: str, defects: List[DefectDetection]) -> None:
        """Store defect annotations in database"""
        try:
            from sqlalchemy import select, delete
            from core.database import get_db
            from models import File, Annotation
            
            async for db in get_db():
                # Find file by path
                file_stmt = select(File).where(File.file_path == str(file_path))
                file_result = await db.execute(file_stmt)
                file_record = file_result.scalar_one_or_none()
                
                if not file_record:
                    print(f"File not found in database: {file_path}")
                    return
                
                # Clear existing defect annotations
                delete_stmt = delete(Annotation).where(
                    Annotation.file_id == file_record.id,
                    Annotation.annotation_type == 'defect_detection'
                )
                await db.execute(delete_stmt)
                
                # Store new annotations
                for defect in defects:
                    annotation = Annotation(
                        file_id=file_record.id,
                        user_id=self.user_id if self.user_id else file_record.user_id,
                        annotation_type='defect_detection',
                        coordinates={
                            'type': 'bounding_box',
                            'x': defect.bounding_box['x'],
                            'y': defect.bounding_box['y'],
                            'width': defect.bounding_box['width'],
                            'height': defect.bounding_box['height']
                        },
                        properties={
                            'defect_type': defect.defect_type,
                            'confidence': defect.confidence,
                            'severity': defect.severity,
                            'description': defect.description
                        },
                        label=f"{defect.defect_type} ({defect.severity})",
                        created_by_ai=True,
                        ai_agent='defect_detector',
                        confidence_score=defect.confidence
                    )
                    db.add(annotation)
                
                await db.commit()
                print(f"Stored {len(defects)} defect annotations for file {file_record.id}")
                break
                
        except Exception as e:
            print(f"Error storing defect annotations: {e}")
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is supported format"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif', '.png'}
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📁 File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:  # Very large file
            print(f"⚠️  Large file detected ({file_size_mb:.1f} MB)")
            if not self.enable_chunking:
                print("💡 Consider enabling TIFF chunking in .env file")
        
        return True
    
    def get_file_info(self, file_path: str) -> Dict:
        """Get comprehensive file information"""
        try:
            with Image.open(file_path) as img:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                info = {
                    'filename': Path(file_path).name,
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size_mb': file_size_mb,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'is_multipage': hasattr(img, 'n_frames') and img.n_frames > 1,
                    'dpi': img.info.get('dpi', (72, 72)),
                    'needs_chunking': False
                }
                
                if info['is_multipage']:
                    info['n_frames'] = img.n_frames
                
                # Determine if chunking is needed - prioritize actual file size
                width, height = img.size
                
                # For compressed images (JPG, PNG), prioritize actual file size
                if file_size_mb <= 15:  # If compressed file is reasonable size, process as whole
                    info['needs_chunking'] = False
                else:
                    # Only chunk if file is large OR extremely high resolution
                    info['needs_chunking'] = (
                        self.enable_chunking and 
                        (file_size_mb > 30 or  # Much higher file size threshold
                         (width > 8000 and height > 8000))  # Only for truly massive resolution
                    )
                
                return info
        except Exception as e:
            raise ValueError(f"Cannot read image file: {str(e)}")
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
        image_bytes = buffer.getvalue()
        file_size_mb = len(image_bytes) / (1024 * 1024)
        
        print(f"📊 Encoded image: {file_size_mb:.2f} MB, Mode: {image.mode}, Size: {image.size}")
        
        if file_size_mb > self.max_file_size_mb:
            print(f"⚠️  Encoded size: {file_size_mb:.2f} MB (limit: {self.max_file_size_mb} MB)")
        
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        print(f"🔐 Base64 string length: {len(base64_str)} characters")
        
        return base64_str
    
    def create_image_message(self, prompt: str, base64_image: str) -> List:
        """Create LangChain message with image content"""
        return [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ])
        ]
    
    def analyze_single_image(self, prompt: str, base64_image: str, max_tokens: int = 1000) -> str:
        """Analyze a single image using LangChain"""
        try:
            # Update LLM max tokens for this request
            self.llm.max_tokens = max_tokens
            
            # Create message with image
            messages = self.create_image_message(prompt, base64_image)
            
            # Invoke LangChain
            response = self.llm.invoke(messages)
            
            return response.content
            
        except Exception as e:
            print(f"❌ Error in LangChain analysis: {str(e)}")
            return f"❌ Error analyzing image: {str(e)}"
    
    async def analyze_single_image_with_context_stream(self, prompt: str, base64_image: str, max_tokens: int = 1000) -> AsyncIterator[str]:
        """Analyze a single image using LangChain with conversation context and streaming"""
        try:
            # Get conversation history
            chat_history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            
            # Create system message
            system_msg = SystemMessage(content=prompt)
            
            # Create image message
            image_msg = HumanMessage(content=[
                {"type": "text", "text": "Please analyze this image based on our conversation context."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
            
            # Combine system message, conversation history, and current image
            messages = [system_msg] + chat_history + [image_msg]
            
            # Set max tokens
            self.llm.max_tokens = max_tokens
            
            # Stream response from LangChain with conversation context
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
            
        except Exception as e:
            print(f"❌ Error in analyze_single_image_with_context_stream: {str(e)}")
            yield f"❌ Error analyzing image with context: {str(e)}"
    
    async def analyze_image_with_chunking_context(self, file_path: str, query: str, analysis_prompt: str) -> AsyncIterator[str]:
        """Analyze large image with chunking and conversation context - with streaming support"""
        try:
            print(f"🧩 Starting chunked analysis with conversation context for: {file_path}")
            
            # Load and chunk the image
            with Image.open(file_path) as img:
                img.load()
                
                # Handle multi-page images
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    print(f"📄 Multi-page image detected: {img.n_frames} pages")
                    best_page = self.select_best_page(img)
                    img.seek(best_page)
                    print(f"✅ Using page {best_page + 1} (largest/most detailed)")
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    print(f"🎨 Converting from {img.mode} to RGB")
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'LA':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Calculate chunks
                chunks = self.tiff_chunker.calculate_chunks(img.size)
                selected_chunks = self.tiff_chunker.get_comprehensive_chunks(chunks, self.max_chunks)
                
                print(f"📊 Processing {len(selected_chunks)} chunks with conversation context")
                
                chunk_results = []
                for i, chunk_info in enumerate(selected_chunks):
                    try:
                        print(f"🔍 Processing chunk {i+1}/{len(selected_chunks)}: {chunk_info['position']}")
                        
                        # Extract chunk
                        chunk_img = self.tiff_chunker.extract_chunk(img, chunk_info)
                        chunk_base64 = self.image_to_base64(chunk_img)
                        
                        # Create enhanced query for this chunk with context
                        chunk_query = self.create_enhanced_chunk_query(query, chunk_info['position'])
                        chunk_prompt = f"{analysis_prompt}\n\nFOCUS AREA: {chunk_info['position']}\nSPECIFIC QUESTION: {chunk_query}"
                        
                        # Analyze chunk with context (collect streaming response)
                        chunk_response = ""
                        async for chunk_part in self.analyze_single_image_with_context_stream(chunk_prompt, chunk_base64, 1500):
                            chunk_response += chunk_part
                        
                        chunk_results.append({
                            'position': chunk_info['position'],
                            'response': chunk_response,
                            'chunk_info': chunk_info
                        })
                        
                    except Exception as e:
                        print(f"❌ Error processing chunk {i+1}: {str(e)}")
                        chunk_results.append({
                            'position': chunk_info['position'],
                            'response': f"Error analyzing this section: {str(e)}",
                            'chunk_info': chunk_info
                        })
                
                # Stream the final combined result (no progress message)
                if self.combine_results and len(chunk_results) > 1:
                    # Stream the intelligently combined result
                    async for chunk in self.combine_chunk_results_intelligently_stream(chunk_results, query):
                        yield chunk
                else:
                    # Stream individual results if combination is disabled
                    yield "📊 SECTION-BY-SECTION ANALYSIS:\n" + "="*50 + "\n\n"
                    for i, result in enumerate(chunk_results):
                        yield f"📍 SECTION {i+1} - {result['position']}:\n"
                        yield f"{result['response']}\n"
                        yield "-" * 40 + "\n\n"
                    
        except Exception as e:
            print(f"❌ Error in chunked analysis with context: {str(e)}")
            yield f"❌ Error processing large image with context: {str(e)}"
    
    def analyze_image_with_chunking(self, file_path: str, query: str) -> str:
        """Analyze large images using intelligent chunking strategy"""
        print("🧩 Starting intelligent chunked analysis...")
        
        with Image.open(file_path) as img:
            img.load()
            
            # Handle multi-page images (MPO, multi-frame TIFF, etc.)
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"📄 Multi-page image detected: {img.n_frames} pages")
                
                # For counting queries, automatically select the largest/most detailed page
                if any(keyword in query.lower() for keyword in ['how many', 'count', 'number of']):
                    print("🔍 Auto-selecting best page for counting analysis...")
                    best_page = self.select_best_page(img)
                    img.seek(best_page)
                    print(f"✅ Using page {best_page + 1} (largest/most detailed)")
                else:
                    # For other queries, use the first page by default
                    print("📋 Using first page for analysis...")
                    img.seek(0)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                print(f"🎨 Converting from {img.mode} to RGB")
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'LA':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Calculate chunks
            chunks = self.tiff_chunker.calculate_chunks(img.size)
            selected_chunks = self.tiff_chunker.get_comprehensive_chunks(chunks, self.max_chunks)
            
            print(f"📊 Analyzing {len(selected_chunks)} strategic chunks from {len(chunks)} total")
            
            chunk_results = []
            
            for i, chunk_info in enumerate(selected_chunks):
                print(f"🔍 Processing chunk {i+1}/{len(selected_chunks)}: {chunk_info['position']}")
                
                # Extract chunk
                chunk_img = self.tiff_chunker.extract_chunk(img, chunk_info)
                base64_chunk = self.image_to_base64(chunk_img)
                
                # Create enhanced query for this chunk
                enhanced_query = self.create_enhanced_chunk_query(query, chunk_info['position'])
                
                # Analyze chunk using LangChain
                try:
                    response_content = self.analyze_single_image(enhanced_query, base64_chunk, 2000)
                    
                    chunk_result = {
                        'chunk_id': chunk_info['id'],
                        'position': chunk_info['position'],
                        'response': response_content
                    }
                    chunk_results.append(chunk_result)
                    
                except Exception as e:
                    print(f"❌ Error analyzing chunk {i+1}: {str(e)}")
                    continue
            
            # Combine results intelligently
            if self.combine_results and len(chunk_results) > 1:
                return self.combine_chunk_results_intelligently(chunk_results, query)
            else:
                return self.format_individual_results(chunk_results)
    
    def create_enhanced_chunk_query(self, original_query: str, position: str) -> str:
        """Create enhanced query for chunk analysis with intelligent context"""
        enhanced_query = f"""{self.master_prompt}

🧩 CHUNK ANALYSIS CONTEXT:
Hi! I'm analyzing a specific section from position: {position} of a larger image. I'm excited to explore what's visible in this particular area!

- This is just one piece of the bigger picture, but I'll give it my full attention! 
- I'll apply the same friendly, intelligent detection and analysis approach
- I'll focus enthusiastically on what's visible in this specific section
- I'll note any partial objects or structures that might extend beyond this view with curiosity
- I'll maintain my warm, conversational tone while being thorough

USER'S QUESTION: {original_query}

Let me dive into this section with enthusiasm and provide you with insights while keeping the same friendly, engaging approach! I'll make sure to ask great follow-up questions about what I discover here! 🔍✨"""
        
        return enhanced_query

    def combine_chunk_results_intelligently(self, chunk_results: List[Dict], original_query: str) -> str:
        """Intelligently combine multiple chunk analysis results"""
        print("🔗 Combining chunk results using intelligent synthesis...")
        
        # Prepare comprehensive summary for intelligent combination
        combination_prompt = f"""{self.master_prompt}

🎯 EXCITING SYNTHESIS TASK:
Hi there! I've just finished analyzing {len(chunk_results)} different sections of your large image, and now I'm thrilled to bring it all together for you! 🎉

My mission is to create a comprehensive, unified analysis that:
1. 🔗 Synthesizes all my findings into one coherent, exciting overview
2. 🤔 Resolves any conflicts or overlapping observations I found
3. 📊 Provides you with accurate counts and measurements across all sections
4. 🌟 Identifies fascinating patterns and relationships across the entire image
5. 💡 Delivers actionable insights based on the complete picture

YOUR ORIGINAL QUESTION: {original_query}

Here's what I discovered in each section:

SECTION ANALYSES:
"""
        
        for i, result in enumerate(chunk_results, 1):
            combination_prompt += f"\n🔍 SECTION {i} ({result['position']}):\n{result['response']}\n"
        
        combination_prompt += f"""

🚀 NOW FOR THE EXCITING SYNTHESIS:
Please help me create a warm, engaging, and comprehensive response that:
- Brings together all these insights with enthusiasm! 🌟
- Eliminates any repetition while keeping all the important details
- Ensures accurate totals and measurements (I want to be precise for you!)
- Highlights the most significant and interesting findings
- Structures everything clearly with friendly explanations
- Includes both summary insights and detailed findings
- Maintains my friendly, conversational tone throughout
- Ends with engaging follow-up questions to help you explore more!

Remember to stay true to my personality - be super friendly, appreciative, and ask great questions! 😊✨"""

        try:
            self.llm.max_tokens = 2500
            response = self.llm.invoke([HumanMessage(content=combination_prompt)])
            combined_result = response.content
            
            return f"🔍 COMPREHENSIVE INTELLIGENT ANALYSIS:\n{combined_result}\n\n📋 DETAILED SECTION BREAKDOWN:\n" + \
                   self.format_individual_results(chunk_results)
                   
        except Exception as e:
            print(f"❌ Error combining results: {str(e)}")
            return self.format_individual_results(chunk_results)
    
    def format_individual_results(self, chunk_results: List[Dict]) -> str:
        """Format individual chunk results"""
        formatted = "📊 SECTION-BY-SECTION ANALYSIS:\n" + "="*50 + "\n\n"
        
        for i, result in enumerate(chunk_results):
            formatted += f"📍 SECTION {i+1} - {result['position']}:\n"
            formatted += f"{result['response']}\n"
            formatted += "-" * 40 + "\n\n"
        
        return formatted
    
    def process_regular_image(self, file_path: str) -> Image.Image:
        """Process regular-sized images with proper MPO/multi-page handling"""
        with Image.open(file_path) as img:
            img.load()
            
            # Handle multi-page images (MPO, multi-frame TIFF, etc.)
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"📄 Multi-page image detected: {img.n_frames} pages")
                
                # Auto-select the best page for analysis
                print("🔍 Auto-selecting best page for analysis...")
                best_page = self.select_best_page(img)
                img.seek(best_page)
                print(f"✅ Using page {best_page + 1} (largest/most detailed)")
            
            processed_img = img.copy()
        
        # Convert to RGB if necessary
        if processed_img.mode != 'RGB':
            print(f"🎨 Converting from {processed_img.mode} to RGB")
            if processed_img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', processed_img.size, (255, 255, 255))
                if processed_img.mode == 'LA':
                    processed_img = processed_img.convert('RGBA')
                background.paste(processed_img, mask=processed_img.split()[-1])
                processed_img = background
            else:
                processed_img = processed_img.convert('RGB')
        
        # Resize if needed
        if (processed_img.size[0] > self.max_image_size or 
            processed_img.size[1] > self.max_image_size):
            processed_img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            print(f"🔄 Resized to: {processed_img.size}")
        
        return processed_img
    
    def select_best_page(self, img) -> int:
        """Select the best page from multi-page image for analysis"""
        best_page = 0
        max_pixels = 0
        
        for i in range(img.n_frames):
            img.seek(i)
            pixels = img.size[0] * img.size[1]
            if pixels > max_pixels:
                max_pixels = pixels
                best_page = i
        
        return best_page
    
    async def analyze_image_stream(self, file_path: str, query: str) -> AsyncIterator[str]:
        """Main function to analyze image with intelligent detection, defect detection, and streaming"""
        try:
            # Validate and get file info
            self.validate_file(file_path)
            file_info = self.get_file_info(file_path)
            
            print(f"📸 Original Image Info: {file_info['size'][0]}x{file_info['size'][1]}, "
                  f"{file_info['format']}, {file_info['file_size_mb']:.1f} MB")
            
            original_img_width, original_img_height = file_info['size']
            
            # Check if this is a defect detection query
            is_defect_query = self._is_defect_detection_query(query)
            
            # Get conversation history for context
            chat_history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            
            # Process the image to get actual dimensions used by AI
            if file_info['needs_chunking']:
                print("🧩 Using chunking strategy for very large image with streaming support")
                # For chunked analysis, use original dimensions since chunks are from original image
                processed_img_width, processed_img_height = original_img_width, original_img_height
                analysis_prompt = self._create_defect_detection_prompt(query, processed_img_width, processed_img_height) if is_defect_query else self._create_intelligent_master_prompt()
                
                async for chunk in self.analyze_image_with_chunking_context(file_path, query, analysis_prompt):
                    yield chunk
            else:
                print("📱 Processing complete image for intelligent analysis with streaming")
                processed_image = self.process_regular_image(file_path)
                processed_img_width, processed_img_height = processed_image.size
                
                print(f"🔄 Processed Image Info: {processed_img_width}x{processed_img_height}")
                print(f"📏 Scale factors: X={processed_img_width/original_img_width:.3f}, Y={processed_img_height/original_img_height:.3f}")
                
                # Create prompt using PROCESSED image dimensions (what AI actually sees)
                if is_defect_query:
                    print("🔍 Defect detection query detected - using specialized prompt with processed dimensions")
                    analysis_prompt = self._create_defect_detection_prompt(query, processed_img_width, processed_img_height)
                else:
                    # Use existing intelligent analysis prompt
                    if chat_history:
                        analysis_prompt = f"""{self.master_prompt}

🎯 CONTINUING OUR CONVERSATION:
Based on our previous conversation, I'll analyze this image while maintaining our conversational context.

YOUR CURRENT QUESTION: {query}

I'll provide insights about this image while remembering our conversation history and maintaining a natural, conversational tone."""
                    else:
                        analysis_prompt = f"""{self.master_prompt}

🎉 EXCITING IMAGE ANALYSIS OPPORTUNITY:
Hi there! Thank you so much for sharing this image with me - I'm absolutely thrilled to analyze it for you! 

YOUR QUESTION: {query}

I'm going to use my intelligent detection framework to explore every fascinating detail in this image. I'll adapt my expertise based on exactly what I observe and provide you with comprehensive, engaging insights that will help you understand everything that's happening here!

Let me dive in with enthusiasm and discover all the amazing things this image has to show us! 🔍✨"""

                base64_image = self.image_to_base64(processed_image)
                
                # Collect full response for defect extraction
                full_response = ""
                
                print(f"🤖 Streaming from {self.model_name} for intelligent analysis with context...")
                
                async for chunk in self.analyze_single_image_with_context_stream(analysis_prompt, base64_image, 2500):
                    full_response += chunk
                    # Yield clean chunk (without defect data markers)
                    if "---DEFECT_DATA_START---" not in chunk:
                        yield chunk
                
                # Process defects if this was a defect detection query
                if is_defect_query:
                    # Extract defects using PROCESSED image dimensions (coordinates from AI)
                    defects = self._extract_defect_data(full_response, processed_img_width, processed_img_height)
                    
                    if defects:
                        # Scale defect coordinates back to ORIGINAL image dimensions for storage
                        # This ensures coordinates are stored relative to the original uploaded image
                        scale_x = original_img_width / processed_img_width
                        scale_y = original_img_height / processed_img_height
                        
                        scaled_defects = []
                        for defect in defects:
                            scaled_bbox = {
                                'x': int(defect.bounding_box['x'] * scale_x),
                                'y': int(defect.bounding_box['y'] * scale_y),
                                'width': int(defect.bounding_box['width'] * scale_x),
                                'height': int(defect.bounding_box['height'] * scale_y)
                            }
                            
                            scaled_defect = DefectDetection(
                                defect_type=defect.defect_type,
                                confidence=defect.confidence,
                                bounding_box=scaled_bbox,
                                description=defect.description,
                                severity=defect.severity
                            )
                            scaled_defects.append(scaled_defect)
                            
                            print(f"🔄 Scaled defect coordinates: {defect.bounding_box} -> {scaled_bbox}")
                        
                        # Store scaled coordinates (relative to original image)
                        await self._store_defect_annotations(file_path, scaled_defects)
                        
                        # Send defect summary
                        yield f"\n\n🎯 **DEFECT DETECTION COMPLETE**: Found {len(scaled_defects)} defect(s)\n"
                        yield "💡 **Red bounding boxes will appear on the image shortly!**\n\n"
                        
                        # Send detailed defect list (using original coordinates)
                        for i, defect in enumerate(scaled_defects, 1):
                            severity_emoji = {
                                'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '⛔'
                            }.get(defect.severity, '⚠️')
                            
                            yield f"**{i}. {defect.defect_type.title()} {severity_emoji}**\n"
                            yield f"   📍 Location: ({defect.bounding_box['x']}, {defect.bounding_box['y']})\n"
                            yield f"   📏 Size: {defect.bounding_box['width']}×{defect.bounding_box['height']} px\n"
                            yield f"   🎯 Confidence: {defect.confidence:.1%}\n"
                            yield f"   📝 {defect.description}\n\n"
                    else:
                        yield "\n\n✅ **No significant defects detected** in this image!\n"
                
                print(f"✅ Completed streaming intelligent analysis")
                
        except Exception as e:
            yield f"❌ Error processing image: {str(e)}"
    
    async def chat_with_context_stream(self, message: str) -> AsyncIterator[str]:
        """Handle general chat with intelligent context, persistent conversation history and streaming"""
        try:
            # Add user message to conversation history
            if self.project_id and self.user_id:
                from .conversation_manager import conversation_manager
                conversation_manager.add_user_message(self.project_id, self.user_id, message)
            
            # Create system message with intelligent context
            system_msg = SystemMessage(content=f"""You are Alex, an enthusiastic and friendly GeoAI specialist! 🌟

{self.master_prompt}

🗣️ CHAT CONVERSATION GUIDELINES:
- Be super friendly, warm, and conversational
- Show genuine interest in what the user is asking
- For GeoAI, imagery, or geospatial questions: Get excited and provide detailed, helpful answers!
- For general questions that relate to your expertise: Answer helpfully while connecting back to your specialty
- For completely off-topic questions: Politely redirect with enthusiasm for your specialty area
- Always ask engaging follow-up questions
- Use appropriate emojis to convey your enthusiasm
- Be encouraging and appreciative of user curiosity

Example redirects for off-topic questions:
"That's a really interesting question! While I specialize in image analysis and GeoAI, I'd absolutely love to help you with any imagery or geospatial questions you might have! What kind of images or aerial data are you working with? 😊🌍"

Remember: Stay friendly, helpful, and always try to guide the conversation toward areas where you can provide the most value! 🚀""")
            
            # Get conversation history from memory
            chat_history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            
            # Create messages list with system message and conversation history
            messages = [system_msg] + chat_history + [HumanMessage(content=message)]
            
            # Use LangChain with streaming
            self.llm.max_tokens = 1500
            
            # Collect response for conversation history
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            # Add AI response to conversation history
            if self.project_id and self.user_id:
                from .conversation_manager import conversation_manager
                conversation_manager.add_ai_message(self.project_id, self.user_id, full_response)
            else:
                # Fallback for basic memory
                if hasattr(self.memory, 'chat_memory'):
                    self.memory.chat_memory.add_user_message(message)
                    self.memory.chat_memory.add_ai_message(full_response)
            
        except Exception as e:
            print(f"Error in chat_with_context_stream: {e}")
            yield f"❌ Oops! I encountered a technical hiccup, but I'm still here and excited to help you with your image analysis questions! Could you try asking again? 😊"
    
    async def combine_chunk_results_intelligently_stream(self, chunk_results: List[Dict], original_query: str) -> AsyncIterator[str]:
        """Stream intelligently combined chunk analysis results"""
        print("🔗 Combining chunk results using intelligent synthesis with streaming...")
        
        # Prepare comprehensive summary for intelligent combination
        combination_prompt = f"""{self.master_prompt}

🎯 EXCITING SYNTHESIS TASK:
Hi there! I've just finished analyzing {len(chunk_results)} different sections of your large image, and now I'm thrilled to bring it all together for you! 🎉

My mission is to create a comprehensive, unified analysis that:
1. 🔗 Synthesizes all my findings into one coherent, exciting overview
2. 🤔 Resolves any conflicts or overlapping observations I found
3. 📊 Provides you with accurate counts and measurements across all sections
4. 🌟 Identifies fascinating patterns and relationships across the entire image
5. 💡 Delivers actionable insights based on the complete picture

YOUR ORIGINAL QUESTION: {original_query}

Here's what I discovered in each section:

SECTION ANALYSES:
"""
        
        for i, result in enumerate(chunk_results, 1):
            combination_prompt += f"\n🔍 SECTION {i} ({result['position']}):\n{result['response']}\n"
        
        combination_prompt += f"""
Please help me create a warm, engaging, and comprehensive response that:
- Brings together all these insights with enthusiasm! 🌟
- Eliminates any repetition while keeping all the important details
- Ensures accurate totals and measurements (I want to be precise for you!)
- Highlights the most significant and interesting findings
- Structures everything clearly with friendly explanations
- Includes both summary insights and detailed findings
- Maintains my friendly, conversational tone throughout
- Ends with engaging follow-up questions to help you explore more!

Remember to stay true to my personality - be super friendly, appreciative, and ask great questions! 😊✨

Provide ONLY the final synthesized analysis - do not include the individual section breakdowns."""

        try:
            self.llm.max_tokens = 2500
            
            # Stream the combined result
            messages = [HumanMessage(content=combination_prompt)]
            
            yield "🔍 COMPREHENSIVE INTELLIGENT ANALYSIS:\n\n"
            
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            print(f"❌ Error combining results: {str(e)}")
            yield "❌ Error combining analysis results. Here are the individual sections:\n\n"
            for i, result in enumerate(chunk_results):
                yield f"📍 SECTION {i+1} - {result['position']}:\n"
                yield f"{result['response']}\n"
                yield "-" * 40 + "\n\n" 