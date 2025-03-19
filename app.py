import os
import streamlit                    as st
from langchain_groq                 import ChatGroq
from langchain.chains               import LLMMathChain, LLMChain
from langchain.prompts              import PromptTemplate
from langchain.agents               import Tool, initialize_agent
from langchain.callbacks            import StreamlitCallbackHandler
from langchain_community.utilities  import WikipediaAPIWrapper
from langchain.agents.agent_types   import AgentType

from dotenv import load_dotenv
load_dotenv()

## Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver and Data Search Assistant")
st.title("Text To Math Problem Solver and Data Search Assistant")