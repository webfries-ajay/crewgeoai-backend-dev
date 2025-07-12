def combine_defect_results(chunk_results, original_query):
    # Wind mills-specific defect combination logic
    # For now, just concatenate all chunk responses (customize as needed)
    combined = '\n'.join([chunk['response'] for chunk in chunk_results])
    return f"[WIND MILLS] Combined Defect Results:\n{combined}" 