from .categories.mining_master_prompt import get_master_prompt_from_env as get_mining_prompt
from .categories.agriculture_master_prompt import get_master_prompt_from_env as get_agriculture_prompt
from .categories.construction_master_prompt import get_master_prompt_from_env as get_construction_prompt
from .categories.solar_master_prompt import get_master_prompt_from_env as get_solar_prompt
from .categories.forestry_master_prompt import get_master_prompt_from_env as get_forestry_prompt
from .categories.urban_planning_master_prompt import get_master_prompt_from_env as get_urban_planning_prompt
from .categories.wind_mills_master_prompt import get_master_prompt_from_env as get_wind_mills_prompt

CATEGORY_PROMPT_MAP = {
    "mining": get_mining_prompt,
    "agriculture": get_agriculture_prompt,
    "construction": get_construction_prompt,
    "solar": get_solar_prompt,
    "forestry": get_forestry_prompt,
    "urban_planning": get_urban_planning_prompt,
    "wind_mills": get_wind_mills_prompt,
}

def get_master_prompt(category: str) -> str:
    """Get master prompt for a specific category"""
    prompt_func = CATEGORY_PROMPT_MAP.get(category, get_agriculture_prompt)
    print(f"Using master prompt: {category}")
    return prompt_func() 