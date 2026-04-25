import os
from openai import OpenAI

# Initialize client if API key is present
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def generate_summary(farm_data, final_score, risk_category, components, overrides_applied):
    """
    Generates a citizen-friendly summary of the credit decision.
    """
    prompt = f"""
    You are an expert agricultural loan officer assistant. Write a short, plain-language paragraph 
    explaining this loan decision to a non-technical bank manager.

    Farm Data:
    - Taluk: {farm_data.get('taluk')}
    - District: {farm_data.get('district')}
    - Declared Crop: {farm_data.get('declared_crop', 'Unknown')}
    
    Technical Metrics:
    - NDVI (Crop Health): {components.get('pred_crop_health', 0):.2f} (0-1 scale)
    - Soil Quality: {components.get('pred_soil_q', 0):.2f}
    - Water Depth: {components.get('pred_water_depth', 0):.2f}m
    
    Final Decision:
    - Credit Score: {final_score}/100
    - Risk Category: {risk_category}
    
    Notes:
    - Override/Guardrails Applied: {overrides_applied}

    Write 2-3 sentences max. Be professional but simple.
    """
    
    if not client:
        # Fallback if no API key is provided
        mock_summary = f"Based on the evaluation of {farm_data.get('taluk')} taluk, the loan is classified as {risk_category} risk with a score of {final_score}. "
        if overrides_applied:
            mock_summary += f"Special considerations were applied due to deterministic rules. "
        mock_summary += f"The physical indicators show a crop health index of {components.get('pred_crop_health', 0):.2f} and water depth of {components.get('pred_water_depth', 0):.2f}m."
        return mock_summary

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"
