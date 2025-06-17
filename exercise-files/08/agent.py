from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from retrievers.vector_search_index import rag_chain
from retrievers.web_search import search_engine_chain

load_dotenv()

# load the LLM

# define the tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

get_word_length("abc")

@tool
def vector_search_query(query):
    return rag_chain.invoke(query)


# create the prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a  powerful assistant but don't know current events"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)
# bind the tool to the LLM

# create the agent

# run the agent
