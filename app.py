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

## Combine tools into a chain
chain = LLMChain(llm = llm, prompt = promptTemplate)

reasoningTool = Tool(
    name = 'Reasoing Tool',
    func = chain.run,
    description = 'Reasoning Tool for answering logic-based and reasoning questions'
)

## Initialize the agents
assistantAgent = initialize_agent(
    tools = [wikipediaTool, calculator, reasoningTool],
    llm = llm, 
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTIION,
    verbose = False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [{
        'role': 'assistant',
        'content': 'Hello! I am an AI assistant. I can help you with math problems and data search. How can I help you today?'
        }
    ]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])
    

## Function to generate a response
def generateResponse(question):
    response = assistantAgent.invoke({'input': question})
    return response

 