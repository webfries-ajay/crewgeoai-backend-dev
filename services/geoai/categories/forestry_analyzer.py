def combine_defect_results(chunk_results, original_query):
    # Forestry-specific defect combination logic
    combined = '\n'.join([chunk['response'] for chunk in chunk_results])
    return f"[FORESTRY] Combined Defect Results:\n{combined}" 