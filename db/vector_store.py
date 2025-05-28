from llm.azure_llm import create_azure_embeddings_llm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
from utils import get_connection

SOURCE_FILE_NAME = "Docs/Business_DOC.txt"
# Mongo
DOCDB_PASSWORD = os.getenv("docdb_password")
DOCDB_DBNAME = os.getenv("docdb_dbname")
DOCDB_USERNAME = os.getenv("docdb_username")

# loader = TextLoader(SOURCE_FILE_NAME)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

with open(SOURCE_FILE_NAME, "r") as file:
    documents = file.read()
file.close()

headers_to_split_on = [
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
docs = markdown_splitter.split_text(documents)

openai_embeddings = create_azure_embeddings_llm()



client = get_connection(username=DOCDB_USERNAME, password=DOCDB_PASSWORD, db_name=DOCDB_DBNAME)
db = client["tutorial"]
collection_name = "orientation"
index_name = "orientation-index"
collection = db[collection_name]

vectorstore = AzureCosmosDBVectorSearch.from_documents(
    docs,
    openai_embeddings,
    collection=collection,
    index_name=index_name,
)

num_lists=100
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.1

vectorstore.create_index(
    num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
)

print("\nVector Store available.\n")