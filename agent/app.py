import streamlit as st
import asyncio
from main import CustomAgentExecutor, llm, prompt, tools
from langchain_core.messages import HumanMessage, AIMessage

st.title("Insurance Agent Assistant")

# For display (only user/agent strings)
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# For LangChain agent (actual message objects)
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

user_input = st.text_input("Ask me about insurance policies:")

# Create executor once
agent_executor = CustomAgentExecutor(llm=llm, prompt=prompt, tools=tools)

if st.button("Send") and user_input.strip():
    # Use agent_history for LangChain
    response = asyncio.run(agent_executor.invoke(user_input, st.session_state.agent_history))

    # Update display history (just strings)
    st.session_state.display_history.append({"user": user_input, "agent": response})

    # Update agent history with proper message objects (your agent already appends them inside invoke)
    # So we can just sync here
    st.session_state.agent_history = agent_executor.chat_history

# Show display history
for chat in st.session_state.display_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Agent:** {chat['agent']}")
    st.markdown("---")
