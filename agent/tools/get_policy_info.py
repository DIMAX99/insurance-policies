from langchain.tools import tool
import json
from llm_huggingface import llm
from langchain_core.prompts import ChatPromptTemplate

# Load policies
with open('final_realistic_insurance_policies.json', 'r') as file:
    policies = json.load(file)

# Create a summary prompt
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert insurance assistant who summarizes policies clearly and concisely for users."),
    ("human", "{policy_info}")
])

@tool
async def get_policy_info_from_dataset(policy_id: str) -> dict:
    """
    Returns a short LLM-generated summary of a specific policy_detail by using policy ID.
    """
    for policy in policies:
        if policy["id"] == policy_id:
            # Convert policy fields to a string or dict
            policy_text = (
                f"Policy Name: {policy.get('name', 'N/A')}\n"
                f"Type: {policy.get('type', 'N/A')}\n"
                f"Premium: ₹{policy.get('premium', 'N/A')}\n"
                f"Sum Insured: ₹{policy.get('sum_insured', 'N/A')}\n"
                f"detailed_terms: {policy.get('detailed_terms', 'N/A')}\n"
                f"Benefits: {policy.get('benefits', 'N/A')}\n"
                f"Company: {policy.get('company', 'N/A')}\n"
            )

            # Prepare the prompt
            full_prompt = await summary_prompt.ainvoke({"policy_info": policy_text})

            # Call the LLM
            summary_text = await llm.ainvoke(full_prompt.to_string())

            return {
                "summary": summary_text,
                "raw_data": policy_text
            }

    return {"error": "Policy not found."}


