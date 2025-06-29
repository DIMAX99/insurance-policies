from langchain.tools import tool
import json

with open('final_realistic_insurance_policies.json', 'r') as file:
    policies = json.load(file)

@tool
def policies_filter(income: float=300000, age: int=25, percentage_to_give: int = 5, family_size: int = 3, policy_type: str = None):
    """Filter policies based on income, age, percentage to give, family size, and optional policy type."""
    solo_filtered_policies = []
    family_filtered_policies = []

    for policy in policies:
        if policy_type is None:
            if family_size == 1:
                if policy['type'] in ['health', 'term_life', 'senior_citizen']:
                    if policy['premium'] > income * percentage_to_give / 100:
                        continue
                    else:
                        solo_filtered_policies.append(policy['id'])
            else:
                if policy['type'] in ['health', 'term_life', 'senior_citizen']:
                    if policy['premium'] > income * percentage_to_give / 100 / family_size:
                        continue
                    else:
                        solo_filtered_policies.append(policy['id'])
                if policy['type'] == 'family_floater':
                    if policy['premium'] > income * percentage_to_give / 100 / family_size:
                        continue
                    else:
                        family_filtered_policies.append(policy['id'])
        else:
            if policy['type'] == policy_type:
                if family_size == 1:
                    if policy['premium'] > income * percentage_to_give / 100:
                        continue
                    else:
                        solo_filtered_policies.append(policy['id'])
                else:
                    if policy['premium'] > income * percentage_to_give / 100 / family_size:
                        continue
                    else:
                        family_filtered_policies.append(policy['id'])

    return {
        "solo_filtered_policies": solo_filtered_policies,
        "family_filtered_policies": family_filtered_policies
    }
