class DecisionValidator:
    """
    Applies deterministic business-logic guardrails to the ML model outputs.
    """
    
    @staticmethod
    def apply_guardrails(repay_prob, risk_category, recommendation, components):
        """
        Adjust predictions based on deterministic rules and confidence calibration.
        """
        final_prob = repay_prob
        final_risk = risk_category
        final_rec = recommendation
        
        # Rule 1 (Override): If predicted water depth > 40m, force High Risk (Reject)
        water_depth = components.get('pred_water_depth', 0)
        if water_depth > 40:
            final_risk = 'High'
            final_rec = 'REJECT - Groundwater depth is dangerously low (>40m), making irrigation highly unreliable.'
            # Cap probability at a low value for High risk
            final_prob = min(final_prob, 0.3)
            
        # Rule 2 (Confidence Calibration): Cross-source agreement
        soil_q = components.get('pred_soil_q', 0)
        crop_health = components.get('pred_crop_health', 0)
        
        # If there's a huge mismatch between soil quality and actual crop health
        # e.g., soil is great but crop is dead, or soil is terrible but crop is thriving (maybe artificial intervention?)
        # we lower the confidence (repayment probability).
        if abs(soil_q - crop_health) > 0.4:
            # Reduce probability by 15%
            final_prob = final_prob * 0.85
            if final_prob < 0.4 and final_risk != 'High':
                final_risk = 'Moderate'
                final_rec = 'REVIEW - Conflicting signals between soil quality and historical crop health.'

        return final_prob, final_risk, final_rec
