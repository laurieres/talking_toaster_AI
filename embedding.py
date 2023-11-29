from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

api_key = "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"

#fridge
loader = PyPDFLoader("/Users/arnauddevoghel/code/arnaud2vo/talking_toaster_AI/datasets/manuals/fridge.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key = api_key)

vector_db = Chroma.from_documents(texts, embeddings)
