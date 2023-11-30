from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI # Choosing which LLM
from langchain.chains.question_answering import load_qa_chain
import os


def communicate_with_manual(object_type, question):
    api_key = os.environ.get("OPENAI_API_KEY")
    manuals_folder = "datasets/manuals"

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

    # Use OpenAI language model
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="map_reduce")

    # Perform similarity search and run the chain
    query = question
    docs = vector_db.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return result

# Example usage:
object_type = "Refrigerator"
question = "How do i make it work?"
result = communicate_with_manual(object_type, question)
print(result)
