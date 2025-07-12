def combine_defect_results(chunk_results, original_query):
    # Urban planning-specific defect combination logic
    combined = '\n'.join([chunk['response'] for chunk in chunk_results])
    return f"[URBAN PLANNING] Combined Defect Results:\n{combined}" 