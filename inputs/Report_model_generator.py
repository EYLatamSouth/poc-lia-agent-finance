from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from llm.azure_llm import create_azure_chat_llm
import pandas as pd
from utils import txt_para_pdf, create_pdf


DOC_NAME = "Docs/Business_DOC_2"

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business analyst of a supermarket.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# Model
llm = create_azure_chat_llm()
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
# Chain
chain = prompt | llm

def call_model(state: MessagesState):
    chain = prompt | llm
    response = chain.invoke(state)
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# Config
config = {"configurable": {"thread_id": "001"}}


query = """
    You need to generate a tutorial guide for the Data Analysts of your team.
    This guide must contain the description of each column of your database.
    A sample of the data is DATA_DICT.

    You also need to generate a report model to the analysts. 
    The model must contain the main analysis they have to do with the data, describing the main insights. Don't include predictions.
    Use # to titles, ## to subtitles and ### to secondaries subtitles. Use only one asterisk to bold words: *text*.

"""

df = pd.read_csv("docs/supermarket_v2.csv")
df3 = df.head(5)
df3 = df3[[column for column in df3.columns if "nnamed" not in column]]
df3 = str(df3.to_dict())
query = query.replace("DATA_DICT",df3)


with open(f"{DOC_NAME}.txt", "w", encoding="utf-8") as file:
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config) 
    res = output["messages"][-1]  # Pegando a última mensagem da LLM
    
    # Exibir a mensagem formatada no terminal
    res.pretty_print()  
    
    # Escrever no arquivo, garantindo a formatação correta
    file.write(res.content + "\n")  
file.close()

create_pdf(filename=f"{DOC_NAME}.pdf", text=res.content + '\n')


