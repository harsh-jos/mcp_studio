import streamlit as st
from mcp_use import MCPAgent, MCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import json

st.set_page_config(page_title="MCP Studio", page_icon="Rb")
st.title("MCP Studio")

# Sidebar for API Key
st.sidebar.title("Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

default_config = '{"mcpServers": {"docs-langchain": {"url": "https://docs.langchain.com/mcp"}}}'
config_str = st.sidebar.text_area("MCP Config", value=default_config, key="mcp_config", height=200)

try:
    config = json.loads(config_str)
except json.JSONDecodeError:
    st.error("Invalid JSON in MCP Config")
    st.stop()

if not gemini_api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the Agent (no caching to avoid event loop issues)
def get_agent(api_key):
    # Initialize MCP Client from the dict
    client = MCPClient.from_dict(config)
    
    # Initialize Gemini Model using LangChain wrapper
    # Using gemini-2.5-flash as requested
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    
    # Initialize MCP Agent
    # We pass the client and the model. 
    agent = MCPAgent(
        llm=model,
        client=client,
    )
    
    return agent

# Async function to handle chat
async def chat_handler(prompt, api_key):
    agent = get_agent(api_key)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call the agent
            response = await agent.run(prompt)
            
            # If response is a string
            if isinstance(response, str):
                full_response = response
                message_placeholder.markdown(full_response)
            else:
                # If it's an object with .content or similar
                full_response = str(response)
                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")

# Main UI Logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something to use MCP"):
    asyncio.run(chat_handler(prompt, gemini_api_key))
