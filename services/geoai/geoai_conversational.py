import os
import sys
import re
from pathlib import Path
from services.geoai.unified_langchain_analyzer import UnifiedLangChainAnalyzer

# Load config from .example
EXAMPLE_PATH = Path(__file__).parent / ".example"
if EXAMPLE_PATH.exists():
    with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip()

def extract_image_path(text):
    """Extract image path from text"""
    match = re.search(r'(\S+\.(?:jpg|jpeg|png|tif|tiff))', text, re.IGNORECASE)
    if match and os.path.exists(match.group(1)):
        return match.group(1)
    return None

def is_detailed_request(text):
    """Check if user wants detailed analysis"""
    keywords = ["detailed", "precise", "in-depth", "comprehensive", "full report", "step by step", "explain thoroughly"]
    return any(k in text.lower() for k in keywords)

# Initialize the intelligent analyzer
analyzer = UnifiedLangChainAnalyzer()
current_image = None

def process_intelligent_analysis(user_input: str):
    """Process user input with intelligent detection and analysis"""
    global analyzer, current_image
    
    # Extract image path if present
    image_path = extract_image_path(user_input)
    if image_path:
        current_image = image_path
        print(f"📸 Image detected: {image_path}")
    
    # Clean question
    question = user_input
    if image_path:
        question = user_input.replace(image_path, "").strip()
        if not question:
            question = "What is in this image? Please provide a comprehensive analysis."
    
    if current_image and os.path.exists(current_image):
        print(f"🔍 Analyzing with intelligent detection system")
        result = analyzer.analyze_image(current_image, question)
        return result
    else:
        return "Please provide a valid image path for analysis."

def chat_with_intelligent_context(message: str):
    """Chat with intelligent context using unified LangChain analyzer"""
    global analyzer
    
    return analyzer.chat_with_context(message)

# Backward compatibility functions (for existing code that might use category-based approach)
def process_with_category(user_input: str, category: str = None):
    """Backward compatibility - now uses intelligent detection regardless of category"""
    print(f"ℹ️ Category '{category}' specified but using intelligent detection system instead")
    return process_intelligent_analysis(user_input)

def chat_with_category_context(message: str, category: str = None):
    """Backward compatibility - now uses intelligent context regardless of category"""
    print(f"ℹ️ Category '{category}' specified but using intelligent context instead")
    return chat_with_intelligent_context(message)

def set_category(category: str):
    """Backward compatibility - intelligent system doesn't use fixed categories"""
    print(f"ℹ️ Category setting '{category}' ignored - using intelligent detection system")
    pass 