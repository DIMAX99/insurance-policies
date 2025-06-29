from langchain_core.tools import tool

@tool
def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user and stop."""
    return {"answer": answer, "tools_used": tools_used}

