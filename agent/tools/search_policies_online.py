from serpapi import GoogleSearch
from langchain.tools import tool

serp_api_key = ""#your_serp_api_key_here

@tool
def search_policies_online(age: int = 20, income: float = 300000, family_size: int = 3, percentage_to_give: int = 5):
    """
    Search for insurance policies on Google using SerpApi based on user info.
    """
    max_premium = income * (percentage_to_give / 100)
    query = f"best health insurance policies in India for age {age}, family size {family_size}, premium under ₹{int(max_premium)}"

    search = GoogleSearch({
        "q": query,
        "location": "India",
        "hl": "en",
        "gl": "in",
        "api_key": serp_api_key
    })

    results = search.get_dict()
    policies = []

    if "organic_results" in results:
        for res in results["organic_results"]:
            policy_info = {
                "title": res.get("title"),
                "link": res.get("link"),
                "snippet": res.get("snippet")
            }
            policies.append(policy_info)

    return {
        "policies": policies,
        "summary": f"Found {len(policies)} trending policies online for age {age}, family size {family_size}, and premium under ₹{int(max_premium)}."
    }
