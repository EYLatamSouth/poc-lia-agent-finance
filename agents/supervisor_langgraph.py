from typing_extensions import TypedDict
from typing_extensions import Annotated
from typing_extensions import TypedDict

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from sqlalchemy import create_engine
from llm.azure_llm import create_azure_chat_llm
import json

memory = MemorySaver()
config = {"configurable": {"thread_id": "2"}} # CONFIG (memory)
################################ BANCOS DE DADOS ################################
db_name = "db/acelerador-analytics-finance.db"
engine = create_engine(f"sqlite:///{db_name}")
db = SQLDatabase(engine)

################################ MODELO ################################
llm = create_azure_chat_llm()

################################ BANCOS DE DADOS ################################
# Query prompt template
with open("inputs/Prompts/prompt_query.txt", "r",  encoding='utf-8') as file:
    query_prompt = file.read()
    query_prompt_template = PromptTemplate.from_template(query_prompt)
file.close()

################################ STATES ################################
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


################################ TOOLS ################################
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "tables_info": "\n".join([db.get_table_info([table_name]) for table_name in db.get_usable_table_names()]),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt, config=config)
    return {"query": result["query"]}


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"], config=config)}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"

        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt, config=config)
    return {"answer": response.content}


tools = [write_query, execute_query, generate_answer]

################################ REACT AGENT ################################

graph = create_react_agent(llm, tools=tools, checkpointer=memory)

################################ MAIN ################################

def analytics_accelerator_function(user_command):
    inputs = {"messages": user_command}
    stream = graph.stream(inputs, stream_mode="values", config=config)
    for s in stream:
        message = s["messages"][-1]
        message_content = message.content
        print(message_content)
        if ("answer" in message_content):
            answer_dict_str = message_content.replace("\n", "\\n")
            answer_dict = json.loads(answer_dict_str)
            return(answer_dict["answer"])
