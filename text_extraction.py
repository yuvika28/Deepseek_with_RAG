# Environment setup
from dotenv import load_dotenv
import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_ollama import ChatOllama, OllamaEmbeddings

from docling.document_converter import DocumentConverter
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
vector_store.index.ntotal, len(chunks)
docs = retriever.invoke('what is revenue for september 2024?')
# docs

# Formatting documents for RAG
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

content = format_docs(docs)
# Setting up the RAG chain
def create_rag_chain(retriever):
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        ### Question: {question} 
        
        ### Context: {context} 
        
        ### Answer:
    """
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
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

# Create RAG chain
rag_chain = create_rag_chain(retriever)


# Questions for retrieval
# question = "How much revenue is there for Google in September 2024?"
question = "What is the net income for this quarter, and what are the key drivers contributing to its increase or decrease?"
# question = "Has the company provided guidance for the next quarter or fiscal year? If so, what are the expected revenue and profit margins?"
# question = "Which technology business segment contributed the most to the company's revenue, and what was the percentage growth in that segment?"
# question = "How has the stock market reacted to this earnings report, and were there any notable comments from the CEO or CFO about future performance?"
# question = "ask your own question"

print(f"Question: {question}")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n" + "-" * 50 + "\n")





























































































































# openai_api_key= "AlzaSyBO80toICBCJ-5d6JUE0yL3sdvHU9UdhTw"
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from uuid import uuid4

# # import the .env file
# from dotenv import load_dotenv
# load_dotenv()

# # configuration
# DATA_PATH = r"data"
# CHROMA_PATH = r"chroma_db"

# embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # initiate the vector store
# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings_model,
#     persist_directory=CHROMA_PATH,
# )

# # loading the PDF document
# loader = PyPDFDirectoryLoader(DATA_PATH)

# raw_documents = loader.load()

# # splitting the document
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False,
# )

# # creating the chunks
# chunks = text_splitter.split_documents(raw_documents)

# # creating unique ID's
# uuids = [str(uuid4()) for _ in range(len(chunks))]

# # adding chunks to vector store
# vector_store.add_documents(documents=chunks, ids=uuids)






