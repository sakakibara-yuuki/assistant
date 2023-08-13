#########
import os
# from langchain.llms import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
gmsh_html = './gmsh/gmsh.html'

# llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# text = "what would be a good company name for a company that makes colorful socks?"
# result = llm.predict(text)
# print(result)

#########
# from langchain.document_loaders import TextLoader
# loader = TextLoader('./gmsh/ch1/0.txt')
# result = loader.load()
# print(result)


#########
# from langchain.document_loaders import TextLoader, DirectoryLoader

# loader = DirectoryLoader('./gmsh/ch1/', glob='*.txt', show_progress=True, use_multithreading=True, loader_cls=TextLoader)

# docs = loader.load()
# print(len(docs))
# print(docs)

#########
# data loader
# from langchain.document_loaders import UnstructuredHTMLLoader
# from langchain.text_splitter import CharacterTextSplitter

# loader = UnstructuredHTMLLoader(gmsh_html)

# data = loader.load()
#########
# embedding model
# from langchain.embeddings import OpenAIEmbeddings

# embeddings_model = OpenAIEmbeddings(openai_api_key)

# embeddings = embeddings_model.embed_documents(["text"])

# embedded_query = embeddings_model.embed_query("query")


#########
# vector store
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./gmsh/ch1/0.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "What gmsh is it?"
docs = db.similarity_search(query)
print(docs[0].page_content)
##########
# Retrieval

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.document_loaders import TextLoader


loader = TextLoader('./gmsh/ch1/0.txt')

