from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI # Choosing which LLM
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv


def embed_and_vectorize_pdf(object_type):
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    manuals_folder = "datasets/"

    # Construct the path to the manual based on the object type
    manual_path = os.path.join(manuals_folder, f"{object_type.lower()}.pdf")

    # Load the manual
    loader = PyPDFLoader(manual_path)
    data = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(data)

    # Embed the text using OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_db = Chroma.from_documents(texts, embeddings)

    return vector_db

def communicate_with_manual(vector_db, question):
    api_key = os.environ.get("OPENAI_API_KEY")
    # Use OpenAI language model
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="map_reduce")

    # Perform similarity search and run the chain
    query = question
    docs = vector_db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    return response

# Example usage:
#object_type = "toaster"
#question = "What age can it be used from?"
#print(os.getcwd())
#vector_db = embed_and_vectorize_pdf(object_type)
#result = communicate_with_manual(vector_db, question)
#print(result)
