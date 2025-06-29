from langchain.tools import tool
from llm_huggingface import llm
import json

# Load policies
with open('final_realistic_insurance_policies.json', 'r') as file:
    policies = json.load(file)

@tool
def load_rejection_reasons(policy_id: str, claim_desc: str = None) -> str:
    """
    Explains possible claim rejection reasons based on policy ID and optional user claim description.
    """

    # Find policy
    policy = next((p for p in policies if p["id"] == policy_id), None)
    if not policy:
        return f"No policy found with ID {policy_id}."

    # Build base policy context
    policy_info = f"""
    Policy Details:
    - Name: {policy.get('name')}
    - Type: {policy.get('type')}
    - Exclusions: {policy.get('exclusions')}
    - Claim Rejection Clauses: {policy.get('claim_rejection_clauses')}
    - Decline Conditions: {policy.get('decline_conditions')}
    - Underwriting Profile: {policy.get('underwriting_profile')}
    - Risk Modifiers: {policy.get('risk_modifiers')}
    - Age Band Pricing: {policy.get('age_band_pricing')}
    - Premium Multipliers: {policy.get('premium_multipliers')}
    """

    if claim_desc:
        # Context when claim description is provided
        context = f"""
        You are an expert insurance policy analyst. Your task is to analyze the provided policy and user
        claim information to determine possible reasons for claim rejection.

        User Claim Description: {claim_desc}

        {policy_info}

        If you can't find any reason, return "No reason found in the policy for claim rejection."

        Please provide a clear explanation of the rejection reasons.
        """
    else:
        # Context when claim description is missing
        context = f"""
        You are an expert insurance policy analyst. Analyze the provided policy to list general reasons
        why claims might be rejected, even though the specific claim description is not provided.

        {policy_info}

        Please provide possible general claim rejection reasons based on this policy.
        """

    # Call LLM
    response = llm.invoke(context)
    return response
