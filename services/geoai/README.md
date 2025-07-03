# Dynamic GeoAI Category System

## Overview

The GeoAI system has been completely redesigned to be **dynamic and category-based**, replacing all hardcoded logic with a flexible, extensible architecture. The system now automatically adapts its analysis approach based on the project category, providing specialized insights for different industries and use cases.

## Architecture

### Core Components

1. **Category-Specific Analyzers** (`categories/`)
   - Each category has its own analyzer class with specialized logic
   - Category-specific keywords, analysis types, and prompts
   - Customizable token limits and processing strategies

2. **Master Prompt Router** (`master_prompt_router.py`)
   - Central routing system for category-specific functionality
   - Dynamic prompt selection and analyzer instantiation
   - Intelligent analysis type determination

3. **Unified LangChain Analyzer** (`unified_langchain_analyzer.py`)
   - **NEW**: Single LangChain implementation replacing dual OpenAI implementations
   - Category-agnostic processing with analyzer injection
   - Maintains all advanced chunking and processing capabilities
   - Unified conversation memory management
   - Production-optimized performance

4. **Smart GeoAI Agent** (`smart_geoai_agent.py`)
   - **UPDATED**: Now uses unified LangChain analyzer
   - Dynamic category switching capabilities
   - Enhanced conversation context with category awareness
   - Backward compatible API

## Supported Categories

### 1. **Mining** (`mining`)
- **Keywords**: mining, excavation, quarry, ore, drilling, blasting, equipment
- **Analysis Types**: Equipment damage, safety violations, environmental issues
- **Specializations**: Equipment assessment, safety evaluation, operational hazards

### 2. **Agriculture** (`agriculture`)
- **Keywords**: crop, field, farm, irrigation, harvest, soil, livestock
- **Analysis Types**: Crop health, irrigation issues, pest damage, equipment status
- **Specializations**: Crop assessment, irrigation analysis, yield estimation

### 3. **Construction** (`construction`)
- **Keywords**: construction, building, site, crane, safety, progress, materials
- **Analysis Types**: Safety violations, progress assessment, quality issues
- **Specializations**: Safety evaluation, progress tracking, equipment analysis

### 4. **Solar Energy** (`solar`)
- **Keywords**: solar panel, photovoltaic, efficiency, shading, maintenance
- **Analysis Types**: Panel defects, shading issues, installation quality
- **Specializations**: Defect detection, shading analysis, performance optimization

### 5. **Forestry** (`forestry`)
- **Keywords**: forest, tree, logging, conservation, species, health
- **Analysis Types**: Forest health, logging assessment, conservation status
- **Specializations**: Health analysis, biodiversity assessment, sustainability

### 6. **Urban Planning** (`urban_planning`)
- **Keywords**: urban, city, infrastructure, traffic, density, development
- **Analysis Types**: Building density, traffic patterns, green space analysis
- **Specializations**: Density analysis, transportation networks, sustainability

### 7. **Wind Energy** (`wind_mills`)
- **Keywords**: wind turbine, blade, defect, maintenance, efficiency
- **Analysis Types**: Defect detection, performance analysis, safety assessment
- **Specializations**: Blade inspection, maintenance planning, efficiency optimization

## 🚀 Unified LangChain Implementation

### Production-Ready Architecture

The system now uses a **single unified LangChain implementation** that replaces the previous dual OpenAI approach:

**Before**: 
- `SmartGeoAIAgent` used OpenAI SDK directly
- `ImageAnalyzer` used separate OpenAI SDK instance
- Duplicate API clients and memory management

**After**: 
- `UnifiedLangChainAnalyzer` handles all OpenAI interactions
- Single LangChain client with optimized resource usage
- Unified conversation memory across all operations
- Enhanced error handling and timeout management

### Key Benefits

- **🎯 Performance**: Reduced API connections and memory usage
- **🔧 Maintainability**: Single implementation to maintain and update
- **🚀 Scalability**: Better resource management for production loads
- **🛡️ Reliability**: Consistent error handling and timeout configurations
- **📊 Monitoring**: Centralized logging and metrics collection

### Backward Compatibility

All existing APIs remain unchanged:
```python
# Same interface, improved implementation
agent = SmartGeoAIAgent(category="agriculture")
response = agent.process_message("Analyze this crop field")
```

## Environment Variables

Each category can have its master prompt customized via environment variables:

```bash
# Category-specific master prompts
MINING_MASTER_PROMPT="Your mining specialist prompt..."
AGRICULTURE_MASTER_PROMPT="Your agriculture specialist prompt..."
CONSTRUCTION_MASTER_PROMPT="Your construction specialist prompt..."
SOLAR_MASTER_PROMPT="Your solar energy specialist prompt..."
FORESTRY_MASTER_PROMPT="Your forestry specialist prompt..."
URBAN_PLANNING_MASTER_PROMPT="Your urban planning specialist prompt..."
WIND_MILLS_MASTER_PROMPT="Your wind energy specialist prompt..."

# Category-specific token limits (optional)
MINING_MAX_TOKENS=1200
AGRICULTURE_MAX_TOKENS=1200
CONSTRUCTION_MAX_TOKENS=1200
SOLAR_MAX_TOKENS=1200
FORESTRY_MAX_TOKENS=1200
URBAN_PLANNING_MAX_TOKENS=1200
WIND_MILLS_MAX_TOKENS=1500
```

## Usage Examples

### Basic Usage

```python
from services.geoai.smart_geoai_agent import SmartGeoAIAgent

# Initialize with specific category
agent = SmartGeoAIAgent(category="mining")

# Set image and analyze
agent.current_image = "path/to/mining_site.jpg"
response = agent.process_message("Analyze safety conditions at this mining site")
```

### Dynamic Category Switching

```python
# Switch categories dynamically
agent.set_category("agriculture")
agent.current_image = "path/to/farm_field.jpg"
response = agent.process_message("Assess crop health in this field")
```

### Category-Specific Analysis

```python
from services.geoai.master_prompt_router import (
    get_category_analyzer, determine_analysis_type, create_category_prompt
)

# Get category-specific analyzer
analyzer = get_category_analyzer("solar")

# Determine analysis type
analysis_type = determine_analysis_type("check for panel defects", "solar")
# Returns: "defect_detection"

# Create specialized prompt
prompt = create_category_prompt("analyze shading issues", "solar")
```

## API Integration

The system integrates seamlessly with the existing API:

```python
# In files.py router
category = db_project.category or "agriculture"
agent = SmartGeoAIAgent(master_prompt=master_prompt, category=category)
agent.current_image = str(file_path)
response = agent.process_message(message)
```

## Key Features

### 1. **Intelligent Query Detection**
- Automatically detects category-specific queries
- Routes to appropriate specialized analysis
- Falls back to general analysis when needed

### 2. **Dynamic Analysis Types**
- Determines optimal analysis approach based on query content
- Category-specific analysis type mapping
- Flexible prompt generation

### 3. **Advanced Chunking Support**
- All category analyzers support large image chunking
- Category-specific chunk result combination
- Specialized defect analysis for wind turbines

### 4. **Extensible Architecture**
- Easy to add new categories
- Simple analyzer class interface
- Configurable via environment variables

## Adding New Categories

### 1. Create Category Analyzer

```python
# services/geoai/categories/new_category_master_prompt.py
import os

def get_master_prompt_from_env() -> str:
    return os.getenv("NEW_CATEGORY_MASTER_PROMPT", "Default prompt...")

class NewCategoryAnalyzer:
    def __init__(self):
        self.category_keywords = ["keyword1", "keyword2"]
        self.analysis_types = ["Type1", "Type2"]
        self.max_tokens = int(os.getenv('NEW_CATEGORY_MAX_TOKENS', 1200))
    
    def is_category_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.category_keywords)
    
    def create_category_prompt(self, original_query: str, position: str = None) -> str:
        # Return specialized prompt for this category
        pass
    
    def get_enhanced_query_for_analysis(self, original_query: str, analysis_type: str = "comprehensive") -> str:
        # Return enhanced query based on analysis type
        pass
```

### 2. Update Master Prompt Router

```python
# Add to master_prompt_router.py
from .categories.new_category_master_prompt import get_master_prompt_from_env as get_new_category_prompt, NewCategoryAnalyzer

CATEGORY_PROMPT_MAP = {
    # ... existing categories
    "new_category": get_new_category_prompt,
}

CATEGORY_ANALYZER_MAP = {
    # ... existing categories
    "new_category": NewCategoryAnalyzer,
}
```

### 3. Update Project Schema

```python
# Add to schemas/project.py ProjectCategory enum
NEW_CATEGORY = "new_category"
```

## Testing

Run the test script to verify the system:

```bash
cd backend
python test_dynamic_geoai.py
```

## Migration from Previous System

The system maintains backward compatibility while providing enhanced functionality:

- **Removed**: All hardcoded wind turbine logic
- **Added**: Dynamic category-based analysis
- **Enhanced**: Flexible prompt system
- **Improved**: Extensible architecture

## Performance Considerations

- Category analyzers are instantiated on-demand
- Master prompts are cached per category
- Efficient query routing and analysis type determination
- Optimized chunking strategies per category

## Error Handling

- Graceful fallback to default category (agriculture)
- Robust error handling in category-specific analysis
- Detailed logging for debugging and monitoring

## Future Enhancements

- Machine learning-based category detection
- Custom category creation via UI
- Advanced analytics and reporting per category
- Integration with external domain-specific APIs 