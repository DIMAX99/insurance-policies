# 🧑‍💼 Insurance Agent Assistant

A smart, interactive insurance assistant that helps users explore and understand insurance policies. Built using a custom agent framework (LangChain-style), Hugging Face LLM, and Streamlit for a simple UI.

---

## 💡 Features

- Ask about policies based on age, income, and family size
- Get detailed explanations for specific policy IDs
- Understand why a claim might be rejected
- View step-by-step log of all reasoning and tool calls
- Interactive Streamlit frontend

---

## ⚙️ Tech Stack

- Python 3.10+
- Streamlit
- Hugging Face LLM (e.g., Mistral or any compatible model)
- LangChain Core concepts (prompt templates, message placeholders)
- Pydantic for tool validation

---
In the real world, an insurance agent:
1️⃣ Understands your background (income, age, family size).
2️⃣ Finds suitable policy options.
3️⃣ Explains policy benefits and premium details.
4️⃣ Helps you understand why a claim might get rejected.
5️⃣ Finally, summarizes everything in simple words.

Your AI agent follows exactly this logical flow, using a combination of an LLM (language model), a custom decision-making framework ("agent executor"), and external tools (like database lookups for policies).
 How it works internally
1️⃣ User inputs a question in Streamlit UI (e.g., "Explain policy P036" or "Find best policies for me").

2️⃣ LLM decides which tool to use first.

It may first call a "filter policies" tool based on income, age, family size.

Then call "get policy info" tool to get more details.

Or, for claims, call "explain claim rejection" tool.

3️⃣ Tools are executed on the backend.

Tools are separate Python functions (like APIs) that simulate a real insurance database or logic.

4️⃣ LLM reads the tool outputs, keeps track of all intermediate results ("scratchpad"), and iterates if needed.

5️⃣ Once enough information is gathered, LLM forms a final human-friendly answer.

6️⃣ Streamlit UI displays the final answer along with step-by-step logs of what the model did.
