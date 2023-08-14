#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
from langchain.document_loaders import (
    TextLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader,
    WebBaseLoader,
    PyPDFLoader,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

class BookShelf:
    ### What do you use DOCUMENT LOADER ???? ########
    loaders = []

    web_loader = WebBaseLoader("https://onelab.info/")
    html_loader = BSHTMLLoader("../reference/gmsh.html")
    pdf_loader = PyPDFLoader("../reference/gmsh_paper_preprint.pdf")
    loaders.append(web_loader)
    loaders.append(html_loader)
    loaders.append(pdf_loader)

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    ### What do you use TEXT SPLITTER ???? ########
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

    ### What do you use EMBEDDINTG ???? ########
    embedding = OpenAIEmbeddings()

    ### What do you use VECTOR STORE ???? ########
    vectordb = Chroma.from_documents(documents=documents,
                                     embedding=embedding)
    print(vectordb)
