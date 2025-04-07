# Environment setup
import streamlit as st
from dotenv import load_dotenv
import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

#Importing required libraries
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_ollama import ChatOllama, OllamaEmbeddings
from docling.document_converter import DocumentConverter

#Convert PDF to Markdown
def load_and_convert_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

source = "E:\\Projectone\\mental_health.pdf"
markdown_content = load_and_convert_document(source)

# Splitting markdown content into chunks
def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return markdown_splitter.split_text(markdown_content)

chunks = get_markdown_splits(markdown_content)

# Embedding and vector store setup
def setup_vector_store(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

vector_store = setup_vector_store(chunks)

# Setup retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

# Formatting documents for RAG
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs]) if docs else ""

# Setting up the RAG chain with fallback to DeepSeek
def create_rag_chain(retriever):
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, generate a response using your own knowledge.
        Answer in bullet points. Make sure your answer is relevant to the question.
        ### Question: {question} 
        
        ### Context: {context} 
        
        ### Answer:
    """
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    def retrieve_and_fallback(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        if not context.strip():  # If no relevant context is found
            context = "No relevant context found in the document. Please generate a response using external knowledge."
        return {"context": context, "question": question}

    chain = (
        retrieve_and_fallback
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain

# Load document
source = "E:\\Projectone\\mental_health.pdf"
markdown_content = load_and_convert_document(source)
chunks = get_markdown_splits(markdown_content)

# Create vector store
vector_store = setup_vector_store(chunks)

# Setup retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

# Create RAG chain with fallback
rag_chain = create_rag_chain(retriever)

# Questions for retrieval
question = "What is water?"

print(f"Question: {question}")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n" + "-" * 50 + "\n")

































































































































































# import boto3
# import tabula
# import faiss
# import os
# import json
# import base64
# import pymupdf
# import numpy as np
# from tqdm import tqdm
# import fitz
# import logging
# from botocore.exceptions import ClientError
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from IPython import display

# logger=logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)

# filepath= "E:\\Projectone\\Sciencepyqs.pdf"
# doc=pymupdf.open(filepath)
# num_pages = len(doc)
        
# for num_pages in range(len(doc)):
#             # Render page to an image
#             page = doc[num_pages]
#             pix = page.get_pixmap(dpi=300)  # High DPI for better OCR accuracy

# image_save_dir="processed_data/processed_images"
# text_save_dir="processed_data/processed_text"
# table_save_dir="processed_data/processed_tables"
# page_images_save_dir="processed_data/processed_page_images"

# chunk_size= 700
# overlap=200

# text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap,length_function=len,)
# items=[]
# for page_num in tqdm(range(num_pages),desc="Processing PDF pages"):
#     page=doc[page_num]
#     text=page.get_text()
#     try:
#         tables=tabula.read_pdf(filepath,pages=page_num+1,multiple_tables=True)
#         if tables:
#             for table_idx, table in enumerate(tables):
#                 table_text="\n".join([" | ".join(map(str, row)) for row in table.values])
                
#                 table_file_name=f"{table_save_dir}/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
#                 os.makedirs(table_save_dir, exist_ok=True)
#                 with open(table_file_name,'w') as f:
#                     f.write(table_text)
                 
#                 table_item= {
#                     "page": page_num,
#                     "type":"table",
#                     "text" : table_text,
#                     "path":table_file_name
#                 }    
#                 items.append(table_item)
#                 text= text.replace(table_text,"")
#     except Exception as e:
#         print(f"Error extracting tables from page{page_num}:{str(e)}")
#      #Step 2: Get all TEXT chunks in the current page and store
#     chunks =text_splitter.split_text(text)
    
#     #Generate an item to add to items
#     for i,chunk in enumerate(chunks):
#         text_file_name=f"{text_save_dir}/{filepath}_text_{page_num}_{i}.txt"
        
#         #If the text folder doesn't exist,create one
#         os.makedirs(text_save_dir,exist_ok=True)
#         with open(filepath,'w') as f:
#             f.write(chunk)
            
#         item={}
#         item["page"]= page_num 
#         item["type"]= "text"              
#         item["text"]= chunk     
#         item["path"]= text_file_name
#         items.append(item)
        
#     # Step 3: Get all the IMAGES in the current page and store
#     images =page.get_images()
#     for idx, image in enumerate(images):
#         #Extract the image data
#         xref=image[0]
#         pix=pymupdf.Pixmap(doc,xref)
#         pix.tobytes("png")
#         #Create the image_name that includes the image path
#         image_name=f"{image_save_dir}/{filepath}_image_{page_num}_{idx}_{xref}.png"
#         #If the image folder doesn't exist, create one
#         os.makedirs(image_save_dir, exist_ok=True)
#         #Save the image
#         pix.save(image_name)
#         #Produce base64 string
#         with open(image_name,'rb') as f:
#             image = base64.b64encode(f.read()).decode('utf8')
        
#         item={}
#         item["page"]= page_num
#         item["type"]= "image"
#         item["path"]= image_name
#         item["image"]= image
#         items.append(item)
# #Save pdf pages as images

# def pdf2imgs(filename, page_images_save_dir):
#     page_images_save_dir = pdf2imgs(filepath,page_images_save_dir)

# for page_num in range(num_pages):
#     page_path=os.path.join(page_images_save_dir, f"page_{page_num:03d}.png")
#     #Produce base64 string
#     with open(image_name,'rb') as f:
#         page_image =base64.b64encode(f.read()).decode('utf8')
    
#     item={}
#     item["page"]= page_num
#     item["type"]= "page"
#     item["path"]= page_path
#     item["image"]= page_image
#     items.append(item)

#....................................................................................................#
# import os
# import base64
# import logging
# import fitz  # PyMuPDF
# import tabula
# from tqdm import tqdm
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Configure logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)

# # Input file and directories
# filepath = "E:\\Projectone\\two_papers_science_output.pdf"

# image_save_dir = "processed_data/processed_images"
# text_save_dir = "processed_data/processed_text"
# table_save_dir = "processed_data/processed_tables"
# page_images_save_dir = "processed_data/processed_page_images"

# # Create directories if not exist
# os.makedirs(image_save_dir, exist_ok=True)
# os.makedirs(text_save_dir, exist_ok=True)
# os.makedirs(table_save_dir, exist_ok=True)
# os.makedirs(page_images_save_dir, exist_ok=True)

# # PDF document
# doc = fitz.open(filepath)
# num_pages = len(doc)

# chunk_size = 700
# overlap = 200

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
# items = []

# for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
#     page = doc[page_num]
#     text = page.get_text()

#     # Extract tables
#     try:
#         tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
#         if tables:
#             for table_idx, table in enumerate(tables):
#                 table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
#                 table_file_name = os.path.join(table_save_dir, f"{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt")
                
#                 with open(table_file_name, 'w', encoding='utf-8') as f:
#                     f.write(table_text)

#                 table_item = {
#                     "page": page_num,
#                     "type": "table",
#                     "text": table_text,
#                     "path": table_file_name,
#                 }
#                 items.append(table_item)
#                 text = text.replace(table_text, "")
#     except Exception as e:
#         logger.error(f"Error extracting tables from page {page_num}: {e}")

#     # Extract text chunks
#     chunks = text_splitter.split_text(text)
#     for i, chunk in enumerate(chunks):
#         text_file_name = os.path.join(text_save_dir, f"{os.path.basename(filepath)}_text_{page_num}_{i}.txt")
        
#         with open(text_file_name, 'w', encoding='utf-8') as f:
#             f.write(chunk)

#         text_item = {
#             "page": page_num,
#             "type": "text",
#             "text": chunk,
#             "path": text_file_name,
#         }
#         items.append(text_item)

#     # Extract images
#     for image_idx, image in enumerate(page.get_images(full=True)):
#         xref = image[0]
#         pix = fitz.Pixmap(doc, xref)
        
#         if pix.n > 4:  # Convert CMYK to RGB
#             pix = fitz.Pixmap(fitz.csRGB, pix)
        
#         image_name = os.path.join(image_save_dir, f"{os.path.basename(filepath)}_image_{page_num}_{image_idx}.png")
#         pix.save(image_name)
        
#         with open(image_name, 'rb') as f:
#             image_data = base64.b64encode(f.read()).decode('utf-8')

#         image_item = {
#             "page": page_num,
#             "type": "image",
#             "path": image_name,
#             "image": image_data,
#         }
#         items.append(image_item)

# # Save PDF pages as images
# for page_num in range(num_pages):
#     pix = doc[page_num].get_pixmap(dpi=300)
#     page_image_name = os.path.join(page_images_save_dir, f"page_{page_num:03d}.png")
#     pix.save(page_image_name)

#     with open(page_image_name, 'rb') as f:
#         page_image_data = base64.b64encode(f.read()).decode('utf-8')

#     page_item = {
#         "page": page_num,
#         "type": "page",
#         "path": page_image_name,
#         "image": page_image_data,
#     }
#     items.append(page_item)

# # Close the document
# doc.close()

# # Output items if needed
# print(f"Processed {len(items)} items.")
   
    
    