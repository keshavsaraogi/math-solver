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

st.set_page_config(page_title="Text To Math Problem Solver and Data Search Assistant")
st.title("Text To Math Problem Solver and Data Search Assistant")

if "GROQ_API_KEY" in st.secrets:
    groqApiKey = st.secrets["GROQ_API_KEY"]
else:
    groqApiKey = os.getenv("GROQ_API_KEY")

if not groqApiKey:
    st.info('Please set the GROQ_API_KEY environment variable (or Streamlit secret) to use this app.')
    st.stop()

llm = ChatGroq(model = "gemma2-9b-it", groq_api_key = groqApiKey)

wikipediaWrapper = WikipediaAPIWrapper()
wikipediaTool = Tool(
    name = "Wikipedia",
    func = wikipediaWrapper.run,
    description = "Search Wikipedia for factual summaries. Do not use for calculations or step-by-step reasoning."
)

mathChain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name="Calculator",
    func=mathChain.run,
    description="Use for numeric calculations on pure mathematical expressions. Return only the numeric result when asked."
)

prompt = """
You are an agent tasked for solving user's mathematical problems.
If the user asks for a numerical value, provide the result directly.
If the user asks for a calculation, logically arrive at the solution and provide a detailed explanation and display the steps in points for the question below.
Question: {question}
Answer: 
"""

promptTemplate = PromptTemplate(
    input_variables = ['question'],
    template = prompt
)

chain = LLMChain(llm = llm, prompt = promptTemplate)

reasoningTool = Tool(
    name='Reasoning Tool',
    func=chain.run,
    description='Use for logic/step-by-step reasoning when no numeric computation is required. Do not use for calculations or factual lookups.'
)

assistantAgent = initialize_agent(
    tools = [wikipediaTool, calculator, reasoningTool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [{
        'role': 'assistant',
        'content': 'Hello! I am an AI assistant. I can help you with math problems and data search. How can I help you today?'
    }]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

userQuestion = st.text_area("Ask me a question", "What is the numerical value of pi to 10 decimal places? Please provide only the numerical result, without any code or explanations.")

if st.button("Find My Answer"):
    if userQuestion:
        with st.spinner("Generating Answer..."):
            st.session_state.messages.append({'role': 'user', 'content': userQuestion})
            st.chat_message('user').write(userQuestion)

            streamlitCallback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = assistantAgent.run(userQuestion, callbacks=[streamlitCallback])
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.write('### Response: ')
            st.success(response)
    else:
        st.warning("Please enter a question to get an answer.")
        