import json
import asyncio
import re
from llm_huggingface import llm
from tools.get_policy_info import get_policy_info_from_dataset
from tools.explain_claim_rejection import load_rejection_reasons
from tools.simple_policies_filter import policies_filter
from tools.search_policies_online import search_policies_online
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

REQUIRED_FIELDS_BY_TOOL = {
    "policies_filter": ["income", "age", "family_size"],
    "load_rejection_reasons": ["policy_id", "claim_desc"],
}

def get_missing_fields(tool_name, extracted_args):
    required = REQUIRED_FIELDS_BY_TOOL.get(tool_name, [])
    return [field for field in required if field not in extracted_args]

def extract_last_tool_json(text: str):
    pattern = (
        r'\{'
        r'[^{}]*"tool_name"\s*:\s*"(?:policies_filter|get_policy_info|load_rejection_reasons|final_answer)"'
        r'[^{}]*\{[^{}]*\}'
        r'[^{}]*\}'
    )
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    last_json_str = matches[-1]
    try:
        parsed = json.loads(last_json_str)
        return parsed
    except json.JSONDecodeError:
        return None

def extract_user_params(text: str):
    params = {}
    income_match = re.search(r'income\s*₹?\s*(\d+)', text, re.IGNORECASE)
    if income_match:
        params["income"] = int(income_match.group(1))
    age_match = re.search(r'age\s*(\d+)', text, re.IGNORECASE)
    if age_match:
        params["age"] = int(age_match.group(1))
    family_size_match = re.search(r'family\s*(?:of\s*)?(\d+)', text, re.IGNORECASE)
    if family_size_match:
        params["family_size"] = int(family_size_match.group(1))
    claim_desc_match = re.search(r'(?:claim\s*(?:desc|description)\s*:|claim\s+about)\s*(.+)', text, re.IGNORECASE)
    if claim_desc_match:
        params["claim_desc"] = claim_desc_match.group(1).strip()
    return params

tools = {
    "get_policy_info": get_policy_info_from_dataset,
    "load_rejection_reasons": load_rejection_reasons,
    "policies_filter": policies_filter,
    "search_policies_online": search_policies_online
}

prompt = ChatPromptTemplate.from_messages([
    ('system', (
        "You are a smart insurance agent assistant. You have access to these tools:\n"
        "- policies_filter: needs income, age, family_size.\n"
        "- get_policy_info: needs policy_id.\n"
        "- load_rejection_reasons: needs policy_id and claim_desc.\n\n"
        "-search_policies_online.\n\n"
        "Your job is to decide which tool to use based on user query and available information.\n\n"
        "Rules:\n"
        "1. Respond with exactly one valid JSON object per turn in this format: {{\"tool_name\": \"...\", \"args\": {{...}}}}.\n"
        "2. Use only one tool at a time and wait for its output before deciding next tool.\n"
        "3. If required arguments are missing, ask the user clearly for those missing fields.\n"
        "4. If the user refuses or skips a required field, proceed using only the given information and do not force.\n"
        "5. Never include explanations, extra text, or multiple JSON objects in a single response.\n"
        "6. If you have enough data to answer immediately, directly respond with final tool and args.\n"
        "7. If no policy_id is given, do not guess. Always run policies_filter first to get list, then use first policy if asked.\n"
        "8. You can handle general non-insurance questions politely if user asks.\n"
        "9. if user explicitly mentions 'search online', 'search web', 'trending', or 'latest', always choose search_policies_online and do not use policies_filter or get_policy_info first."
        "10.- search_policies_online: Use this when the user explicitly says 'search web','search online', 'trending', or 'latest'. This tool does NOT require a policy_id. You may pass keywords, or optionally income, age, and family_size if given."
        "Strictly follow these instructions."
    )),
    MessagesPlaceholder(variable_name="summary"),
    ('human', "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who summarizes insurance agent conversations shortly."),
    ("human", "{text}"),
])

class CustomAgentExecutor:
    def __init__(self, llm, tools, prompt, max_iterations=5):
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.chat_history = []
        self.summary_memory = []
        self.selected_policies = []

    async def summarize_interaction(self, new_text):
        combined_text = "\n".join(self.summary_memory) + "\n" + new_text
        full_prompt = await summarize_prompt.ainvoke({"text": combined_text})
        summary = self.llm.invoke(full_prompt.to_string())
        self.summary_memory = [summary]
        return summary

    async def invoke(self, query):
        count = 0
        final_answer = ""
        sratchpad = []
        user_params = extract_user_params(query)

        while count < self.max_iterations:
            full_prompt = await self.prompt.ainvoke({
                "input": query,
                "summary": self.summary_memory,
                "agent_scratchpad": sratchpad
            })
            llm_response = await self.llm.ainvoke(full_prompt.to_string())
            print("LLM Response:", llm_response)

            response = extract_last_tool_json(llm_response)
            if not response:
                return {"message": "Could not extract valid tool JSON from LLM response."}

            toolname = response.get("tool_name")
            args = response.get("args", {})
            print(f"Extracted tool: {toolname}, args: {args}")

            missing = get_missing_fields(toolname, args)
            if missing:
                return {
                    "message": f"Please provide the following missing fields: {missing}",
                    "missing_fields": missing
                }

            if toolname == "final_answer":
                final_answer = args.get("answer", "")
                break

            tool_func = self.tools.get(toolname)
            results = await tool_func.ainvoke(args)
            print(f"Tool {toolname} returned: {results}")

            if toolname == "policies_filter":
                self.selected_policies = results.get("family_filtered_policies", []) or results.get("solo_filtered_policies", [])

                if "first policy" in query.lower() and self.selected_policies:
                    policy_id = self.selected_policies[0]
                    next_tool_response = {"tool_name": "get_policy_info", "args": {"policy_id": policy_id}}
                    print(f"Auto-selecting first policy: {policy_id}")
                    results = await self.tools["get_policy_info"].ainvoke(next_tool_response["args"])
                    print(f"Tool get_policy_info returned: {results}")

            await self.summarize_interaction(str(results))
            sratchpad.append(AIMessage(content=json.dumps(response)))
            sratchpad.append(HumanMessage(content=str(results)))

            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=str(results)))

            if "policies" in results:
                formatted_policies = "\n".join(
                    [f"- {p.get('title')} ({p.get('link')})" for p in results["policies"]]
                )
                final_answer = f"{results.get('summary')}\n\n{formatted_policies}"
            else:
                final_answer = results.get("summary") or results.get("raw_data") or str(results)

            count += 1
            break

        return {
            "final_answer": final_answer,
            "summary": self.summary_memory,
        }

agent_executor = CustomAgentExecutor(llm, tools, prompt)

query = "hi my name is john"

final_response = asyncio.run(agent_executor.invoke(query))
print("✅ Final Answer:", final_response)
