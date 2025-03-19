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

groqApiKey = os.getenv("GROQ_API_KEY")

if not groqApiKey:
    st.info('Please set the GROQ_API_KEY environment variable to use this app.')
    
llm = ChatGroq(model = "gemma2-9b-it", groq_api_key = groqApiKey)

## Initialize wikipedia tool
wikipediaWrapper = WikipediaAPIWrapper()
wikipediaTool = Tool(
    name = "Wikipedia",
    function = wikipediaWrapper.search,
    description = "Search and Solve Wikipedia for information on the topics mentioned"
)

## Initialize the math tool
mathChain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = "Calculator",
    function = mathChain.run,
    description = "Tool for answering math problems. Only mathematically expressions are supported."
)

prompt = """
You are an agent tasked for solving user's mathematical problems.
Logically arrive at the solution and provide a detailed explanation and display the steps in points for the question below.
Question: {question}
Answer: 
"""

promptTemplate = PromptTemplate(
    input_variables = ['questions'],
    template = prompt
)

