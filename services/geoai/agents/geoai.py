import os
import base64
from PIL import Image
import io
from typing import Optional, Tuple, List, Dict
import openai
from pathlib import Path
import math
from dotenv import load_dotenv
import json
from backend.services.geoai.categories import (
    agriculture_analyzer,
    construction_analyzer,
    forestry_analyzer,
    mining_analyzer,
    solar_analyzer,
    urban_planning_analyzer,
    wind_mills_analyzer
)

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

    def get_strategic_chunks(self, chunks: List[Dict], max_chunks: int = 9) -> List[Dict]:
        """Legacy method - now redirects to comprehensive approach"""
        return self.get_comprehensive_chunks(chunks, max_chunks)

class ImageAnalyzer:
    def __init__(self, category: str = "agriculture"):
        """Initialize the Image Analyzer with category-specific settings"""
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', "gpt-4o")  # Default to gpt-4o for superior vision
        self.category = category
        
        # Load settings from .env with category-specific defaults
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
        
        print(f"✅ ImageAnalyzer initialized with model: {self.model}")
        print(f"🏷️ Category: {self.category}")
        print(f"🔧 Settings: {self.max_chunks} chunks, {chunk_overlap}px overlap, {chunk_size}px chunks")
    
    def set_category(self, category: str):
        """Set the analysis category"""
        self.category = category
        print(f"🏷️ Category updated to: {self.category}")
    
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
    
    def process_large_tiff(self, file_path: str, query: str, category_analyzer) -> str:
        """Process large TIFF files using enhanced chunking strategy with category-specific analysis"""
        print("🔄 Processing large image with category-specific analysis...")
        
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
            
            # Calculate chunks with enhanced strategy
            chunks = self.tiff_chunker.calculate_chunks(img.size)
            strategic_chunks = self.tiff_chunker.get_comprehensive_chunks(chunks, self.max_chunks)
            
            print(f"🎯 Analyzing {len(strategic_chunks)} chunks for {self.category} analysis...")
            
            chunk_results = []
            
            for i, chunk_info in enumerate(strategic_chunks):
                print(f"🔍 Processing chunk {i+1}/{len(strategic_chunks)}: {chunk_info['position']}")
                
                # Extract chunk
                chunk_img = self.tiff_chunker.extract_chunk(img, chunk_info)
                
                # Convert to base64
                base64_chunk = self.image_to_base64(chunk_img)
                
                # Create category-specific enhanced query
                enhanced_query = self.create_enhanced_object_query(query, chunk_info['position'], category_analyzer)
                
                # Get category-specific max tokens
                max_tokens = category_analyzer.max_tokens
                
                # Analyze chunk
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": enhanced_query},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_chunk}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )
                    
                    chunk_result = {
                        'chunk_id': chunk_info['id'],
                        'position': chunk_info['position'],
                        'response': response.choices[0].message.content
                    }
                    chunk_results.append(chunk_result)
                    
                except Exception as e:
                    print(f"❌ Error analyzing chunk {i+1}: {str(e)}")
                    continue
            
            # Smart result combination based on category
            if self.combine_results and len(chunk_results) > 1:
                return self.combine_category_results(chunk_results, query, category_analyzer)
            else:
                return self.format_individual_results(chunk_results)
    
    def create_enhanced_object_query(self, original_query: str, position: str, category_analyzer) -> str:
        """Create enhanced query optimized for category-specific analysis"""
        # Check if this is a category-specific query
        is_category_query = category_analyzer.is_category_query(original_query)
        
        if is_category_query:
            return category_analyzer.create_category_prompt(original_query, position)
        
        # Extract what we're counting from the query
        counting_keywords = ['how many', 'count', 'number of']
        summary_keywords = ['describe', 'summary', 'what do you see', 'what is in', 'analyze']
        
        is_counting_query = any(keyword in original_query.lower() for keyword in counting_keywords)
        is_summary_query = any(keyword in original_query.lower() for keyword in summary_keywords)
        
        if is_counting_query:
            # Check if it's a vehicle counting query for enhanced detection
            vehicle_keywords = ['car', 'vehicle', 'auto', 'truck', 'van', 'motorcycle']
            is_vehicle_query = any(keyword in original_query.lower() for keyword in vehicle_keywords)
            
            if is_vehicle_query:
                return f"""This is a section from a larger aerial/road image (position: {position}). 

TASK: {original_query}

VEHICLE DETECTION INSTRUCTIONS:
- Examine this section extremely carefully for ALL types of vehicles
- Look for cars, trucks, vans, motorcycles, buses on roads, parking areas, and shoulders
- Pay special attention to vehicles that blend with road color (gray, dark vehicles)
- Look for vehicle characteristics: rectangular shapes, shadows, reflections, wheel patterns
- Include vehicles that are partially visible at the edges of this section
- Note vehicle colors: white, black, gray, silver, dark blue, etc.
- Look for vehicles in traffic lanes, shoulders, parking spots, side roads

COUNT FORMAT: Provide the exact number and describe what you see:
- Total count: X vehicles
- Colors observed: (white: X, dark/gray: X, other colors: X)
- Types: (cars: X, trucks: X, etc.)
- Locations: (on main road, side areas, etc.)

Be extremely thorough - some vehicles may be small or camouflaged against pavement."""
            else:
                return f"""This is a section from a larger image (position: {position}). 

TASK: {original_query}

IMPORTANT INSTRUCTIONS:
- Look very carefully and thoroughly examine the entire section
- Count ALL instances of the requested objects, including small ones
- Pay attention to objects that might be partially hidden or blend with surroundings
- Include partially visible objects at the edges of this section
- Look for objects in all areas: foreground, background, corners, edges
- Be extremely thorough and accurate in your counting

Please provide ONLY the count number and briefly mention what you see. Keep it concise."""
        
        elif is_summary_query:
            return f"""This is a section from a larger image (position: {position}). 

TASK: {original_query}

IMPORTANT INSTRUCTIONS:
- Describe what you see in this section in detail
- Include information about objects, landscape, buildings, vehicles, people, etc.
- Note colors, textures, and spatial relationships
- Mention the setting/environment (urban, rural, aerial view, etc.)
- Be thorough but concise

Please provide a detailed description of this section."""
        
        else:
            return f"This is a section from a larger image (position: {position}). {original_query}"
    
    def combine_category_results(self, chunk_results: List[Dict], original_query: str, category_analyzer) -> str:
        """Combine chunk results using category-specific logic"""
        print(f"🔗 Combining {self.category} analysis results...")
        
        # Use category-specific defect combination if available
        if hasattr(category_analyzer, 'combine_defect_results'):
            return category_analyzer.combine_defect_results(chunk_results, original_query)
        
        # Use enhanced counting logic for any counting query
        counting_keywords = ['how many', 'count', 'number of']
        summary_keywords = ['describe', 'summary', 'what do you see', 'what is in', 'analyze']
        
        is_counting_query = any(keyword in original_query.lower() for keyword in counting_keywords)
        is_summary_query = any(keyword in original_query.lower() for keyword in summary_keywords)
        
        if is_counting_query:
            return self.combine_counting_results(chunk_results, original_query)
        elif is_summary_query:
            return self.combine_summary_results(chunk_results, original_query)
        else:
            return self.combine_chunk_results(chunk_results, original_query)
    
    def combine_counting_results(self, chunk_results: List[Dict], original_query: str) -> str:
        """Smart combination for any object counting with clean results"""
        print("🔄 Analyzing results from all sections...")
        
        # Check if this is a vehicle query for enhanced analysis
        vehicle_keywords = ['car', 'vehicle', 'auto', 'truck', 'van', 'motorcycle']
        is_vehicle_query = any(keyword in original_query.lower() for keyword in vehicle_keywords)
        
        # Extract counts from each chunk
        section_counts = []
        vehicle_details = []
        import re
        
        for result in chunk_results:
            response = result['response']
            position = result['position']
            
            # Try to extract numbers from response
            numbers = re.findall(r'\b(\d+)\b', response)
            
            # Look for "no" or "zero" indicators
            if any(word in response.lower() for word in ['no ', 'zero', 'none', 'not found', 'not visible']):
                count = 0
            elif numbers:
                # Take the first reasonable number found
                count = int(numbers[0])
            else:
                count = 0
            
            section_counts.append({
                'position': position,
                'count': count,
                'description': response.strip()
            })
            
            # Extract vehicle details if it's a vehicle query
            if is_vehicle_query and count > 0:
                vehicle_details.append({
                    'position': position,
                    'count': count,
                    'details': response.strip()
                })
        
        # Calculate total
        total_detected = sum(sc['count'] for sc in section_counts)
        
        if is_vehicle_query:
            # Enhanced vehicle analysis
            summary_text = f"""I analyzed {len(chunk_results)} overlapping sections of an aerial/road image for vehicle counting: "{original_query}"

DETAILED VEHICLE ANALYSIS:
"""
            for detail in vehicle_details:
                summary_text += f"- {detail['position']}: {detail['count']} vehicles\n  Details: {detail['details']}\n"
            
            summary_text += f"""
Raw total detected: {total_detected} vehicles

ALL SECTION RESPONSES:
"""
            for sc in section_counts:
                summary_text += f"{sc['position']}: {sc['description']}\n"
            
            summary_text += f"""
Please provide a COMPREHENSIVE VEHICLE ANALYSIS including:
1. FINAL ACCURATE COUNT (accounting for potential overlaps between sections)
2. VEHICLE COLOR BREAKDOWN (white cars, dark/gray cars, other colors)
3. VEHICLE TYPES observed (cars, trucks, vans, etc.)
4. CONFIDENCE LEVEL in the count
5. Any vehicles that might have been missed or double-counted

Keep it organized and informative."""
        else:
            # Standard counting analysis
            summary_text = f"""I analyzed {len(chunk_results)} overlapping sections of a large image to answer: "{original_query}"

Section counts:
"""
            for sc in section_counts:
                summary_text += f"- {sc['position']}: {sc['count']}\n"
            
            summary_text += f"""
Raw total: {total_detected}

Section details:
"""
            for sc in section_counts:
                summary_text += f"{sc['position']}: {sc['description']}\n"
            
            summary_text += f"""
Please provide the FINAL ACCURATE COUNT, accounting for any potential overlaps between sections. Give ONLY:
1. The final number
2. One sentence explanation
Keep it very concise and direct."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": summary_text
                    }
                ],
                max_tokens=500 if is_vehicle_query else 300
            )
            
            final_result = response.choices[0].message.content.strip()
            
            # Return enhanced result for vehicles or clean result for others
            if is_vehicle_query:
                return f"🚗 COMPREHENSIVE VEHICLE ANALYSIS:\n{final_result}\n\n📊 Section Breakdown:\n" + \
                       '\n'.join([f"• {sc['position']}: {sc['count']} vehicles" for sc in section_counts])
            else:
                return f"🎯 FINAL ANSWER:\n{final_result}\n\n📊 Section Breakdown:\n" + \
                       '\n'.join([f"• {sc['position']}: {sc['count']}" for sc in section_counts])
                   
        except Exception as e:
            print(f"❌ Error in final analysis: {str(e)}")
            return f"🎯 TOTAL COUNT: {total_detected}\n\n📊 Section Breakdown:\n" + \
                   '\n'.join([f"• {sc['position']}: {sc['count']}" for sc in section_counts])
    
    def combine_chunk_results(self, chunk_results: List[Dict], original_query: str) -> str:
        """Combine multiple chunk analysis results into a cohesive answer"""
        print("🔗 Combining chunk results...")
        
        # Prepare summary for GPT
        summary_text = f"Original question: {original_query}\n\n"
        summary_text += "Analysis of different sections of a large image:\n\n"
        
        for result in chunk_results:
            summary_text += f"Section {result['position']}:\n{result['response']}\n\n"
        
        summary_text += f"Based on the analysis of these {len(chunk_results)} sections, please provide a comprehensive answer to the original question."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": summary_text
                    }
                ],
                max_tokens=1000
            )
            
            combined_result = response.choices[0].message.content
            
            return f"🔍 COMPREHENSIVE ANALYSIS:\n{combined_result}\n\n📋 DETAILED SECTION RESULTS:\n" + \
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
    
    def analyze_image(self, file_path: str, query: str, category: str) -> str:
        """Main function to analyze image with category-specific LLM analysis"""
        # Select the correct analyzer for the category
        if category == "agriculture":
            category_analyzer = agriculture_analyzer
        elif category == "construction":
            category_analyzer = construction_analyzer
        elif category == "forestry":
            category_analyzer = forestry_analyzer
        elif category == "mining":
            category_analyzer = mining_analyzer
        elif category == "solar":
            category_analyzer = solar_analyzer
        elif category == "urban_planning":
            category_analyzer = urban_planning_analyzer
        elif category == "wind_mills":
            category_analyzer = wind_mills_analyzer
        else:
            category_analyzer = None

        if category_analyzer is None:
            return f"❌ Category '{category}' not found or does not have an analyzer."

        try:
            # Validate and get file info
            self.validate_file(file_path)
            file_info = self.get_file_info(file_path)
            
            print(f"📸 Image Info: {file_info['size'][0]}x{file_info['size'][1]}, "
                  f"{file_info['format']}, {file_info['file_size_mb']:.1f} MB")
            
            # Decide processing strategy
            if file_info['needs_chunking']:
                print("🧩 Using chunking strategy for very large image")
                return self.process_large_tiff(file_path, query, category_analyzer)
            else:
                print("📱 Processing complete image for best accuracy")
                processed_image = self.process_regular_image(file_path)
                base64_image = self.image_to_base64(processed_image)
                
                # Create category-specific enhanced query
                is_category_query = category_analyzer.is_category_query(query)
                vehicle_keywords = ['car', 'vehicle', 'auto', 'truck', 'van', 'motorcycle']
                is_vehicle_query = any(keyword in query.lower() for keyword in vehicle_keywords)
                
                if is_category_query:
                    enhanced_query = category_analyzer.create_category_prompt(query)
                    max_tokens = category_analyzer.max_tokens
                elif is_vehicle_query:
                    enhanced_query = f"""AERIAL/ROAD IMAGE VEHICLE ANALYSIS

TASK: {query}

COMPREHENSIVE VEHICLE DETECTION INSTRUCTIONS:
- Examine this COMPLETE aerial/road image very carefully for ALL vehicles
- Look for cars, trucks, vans, motorcycles, buses on all roads, lanes, shoulders, and parking areas
- Pay special attention to vehicles that blend with road color (gray, dark vehicles)
- Look for vehicle characteristics: rectangular shapes, shadows, reflections, wheel patterns
- Note vehicle colors precisely: white, black, gray, silver, dark blue, red, etc.
- Check all traffic lanes, shoulders, side roads, parking spots, intersections

REQUIRED RESPONSE FORMAT:
1. TOTAL COUNT: Exact number of vehicles
2. COLOR BREAKDOWN: white: X, dark/gray: X, other colors: X
3. VEHICLE TYPES: cars: X, trucks: X, vans: X, etc.
4. LOCATIONS: main road lanes, shoulders, side areas, etc.
5. CONFIDENCE: High/Medium/Low and explanation

Be extremely thorough - count every single vehicle visible in the image."""
                    max_tokens = 800
                else:
                    enhanced_query = query
                    max_tokens = 800
                
                # Send to OpenAI
                print(f"🤖 Sending to {self.model} with {max_tokens} max tokens...")
                print(f"📝 Query type: {category.title()} {'Category-Specific' if is_category_query else 'Standard'} Analysis")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": enhanced_query},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=max_tokens
                )
                
                print(f"✅ Received response from AI ({len(response.choices[0].message.content)} characters)")
                return response.choices[0].message.content
                
        except Exception as e:
            return f"❌ Error processing image: {str(e)}"

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

    def combine_summary_results(self, chunk_results: List[Dict], original_query: str) -> str:
        """Combine section descriptions into a comprehensive image summary"""
        print("🖼️ Creating comprehensive image summary...")
        
        # Prepare comprehensive summary
        summary_text = f"""I analyzed {len(chunk_results)} sections of a large image to answer: "{original_query}"

Section descriptions:
"""
        
        for result in chunk_results:
            summary_text += f"\n{result['position']}:\n{result['response']}\n"
        
        summary_text += f"""
Based on these {len(chunk_results)} detailed section analyses, please provide a comprehensive description of the entire image including:
1. Overall scene/setting
2. Main objects and features
3. Layout and spatial relationships
4. Notable details and characteristics

Keep it well-organized and informative but concise."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": summary_text
                    }
                ],
                max_tokens=600
            )
            
            comprehensive_summary = response.choices[0].message.content.strip()
            
            return f"🖼️ COMPREHENSIVE IMAGE ANALYSIS:\n{comprehensive_summary}\n\n📋 Section Details:\n" + \
                   '\n'.join([f"📍 {result['position']}: {result['response'][:100]}..." 
                             if len(result['response']) > 100 
                             else f"📍 {result['position']}: {result['response']}" 
                             for result in chunk_results])
                   
        except Exception as e:
            print(f"❌ Error creating summary: {str(e)}")
            return "🖼️ IMAGE SUMMARY:\n" + \
                   '\n'.join([f"📍 {result['position']}: {result['response']}" for result in chunk_results])