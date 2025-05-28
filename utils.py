from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain_community.vectorstores.azuresearch import AzureSearch
from typing import List
from llm.azure_llm import create_azure_embeddings_llm
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.units import inch
import re
from langchain_community.utilities import SQLDatabase


def create_pdf(filename, text):
    """Converte markdown simples em elementos formatados para PDF."""
    lines = text.split("\n")
    formatted_elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = styles["Heading1"]
    subsubtitle_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    for line in lines:
        line = line.strip()
        
        if line.startswith("### "):  # Subtítulo menor
            formatted_elements.append(Paragraph(line[4:], subsubtitle_style))
        elif line.startswith("## "):  # Subtítulo
            formatted_elements.append(Paragraph(line[3:], subtitle_style))
        elif line.startswith("# "):  # Título principal
            formatted_elements.append(Paragraph(line[2:], title_style))
        elif "**" in line:  # Negrito
            line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)  # Converte **texto** para <b>texto</b>
            formatted_elements.append(Paragraph(line, normal_style))
        else:  # Texto normal
            formatted_elements.append(Paragraph(line, normal_style))
        
        formatted_elements.append(Spacer(1, 0.2 * inch))


    """Cria um PDF formatado a partir de texto markdown-like."""
    doc = SimpleDocTemplate(filename, pagesize=A4)
    doc.build(formatted_elements)
    print("PDF generated.")



def txt_para_pdf(arquivo_txt, arquivo_pdf):
    # Criar o documento PDF
    doc = SimpleDocTemplate(arquivo_pdf, pagesize=A4)
    estilos = getSampleStyleSheet()
    
    estilo_titulo = ParagraphStyle(
        "Titulo", parent=estilos["Heading1"], spaceAfter=12
    )
    estilo_texto = estilos["BodyText"]

    elementos = []

    with open(arquivo_txt, "r", encoding="utf-8") as file:
        for linha in file:
            linha = linha.strip()

            # Identificar títulos no formato ###
            if linha.startswith("###"):
                titulo = linha[3:].strip()
                elementos.append(Paragraph(f"<b>{titulo}</b>", estilo_titulo))
                elementos.append(Spacer(1, 12))
            else:
                # Substituir **texto** por <b>texto</b> para negrito
                linha_formatada = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', linha)
                elementos.append(Paragraph(linha_formatada, estilo_texto))
                elementos.append(Spacer(1, 6))

    doc.build(elementos)
    print(f"PDF gerado com sucesso: {arquivo_pdf}")


## MONGO DB
def save_conversation(client, conversation):
    db = client['TimeCodeBot']
    collection = db['conversations']
    result = collection.insert_one(conversation)
    print(f"New conversation inserted with the following id: {result.inserted_id}")



def get_connection(username, password, db_name):
    """
    Estabelece uma conexão com o DocumentDB usando as credenciais fornecidas.

    Returns:
        pymongo.MongoClient: Uma instância do cliente MongoDB configurada com as credenciais e SSL.
    """

    # "connectionString": "mongodb+srv://<user>:<password>@db-to-vectorstore.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    url = f"mongodb+srv://{username}:{password}@{db_name}.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    try:
        # logger.info("Connecting to CosmoDB...")
        client = MongoClient(url)
        print("Successful!")
        # logger.info("Successful!")
    except Exception as e:
        print(e)
        # logger.error(f"MongoClient: Error connecting to CosmoDB: {e}")
    return client

def databases_markers(databases:list):
    """
    Recebe a lista de bancos de dados selecionados e gera o marcador para o modelo de LLM.
    """
    marker = "@"
    for db in databases:
        db = db.replace("-","_")
        marker += f"{db}+"
    return marker[:-1]

def get_column_names(db: SQLDatabase, table_name: str):
    table_info = db.get_table_info([table_name])
    pattern = r"\(\s*((?:.|\n)+?)\s*\)"
    match = re.search(pattern, table_info)
    
    if match:
        columns_block = match.group(1)
        columns = [line.strip().split()[0] for line in columns_block.split(",")]
        return columns
    else:
        return []
    

def format_markdown_output(text):
    # Adiciona uma quebra dupla depois de cada item numerado
    return re.sub(r'(\d\.\s\*\*.+?\*\*:)', r'\1\n', text)