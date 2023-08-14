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

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


input_list = [
    "https://onelab.info/",
    "../reference/gmsh.html",
    "../reference/gmsh_paper_preprint.pdf",
]


class BookShelf:
    def __init__(self,
                 input_list: list,
                 is_new: bool = False):

        loaders = []
        loaders.append(WebBaseLoader(input_list[0]))
        loaders.append(BSHTMLLoader(input_list[1]))
        loaders.append(PyPDFLoader(input_list[2]))

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        ### What do you use EMBEDDINTG ???? ########
        self.embedding = OpenAIEmbeddings()

        if is_new:
            self.vectordb = self.create_db(docs)
        else:
            self.vectordb = self.load_db()

    def create_db(self, docs):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)

        ### What do you use VECTOR STORE ???? ########
        vectordb = Chroma.from_documents(documents=documents,
                                         embedding=self.embedding,
                                         persist_directory="./db")
        return vectordb

    def load_db(self):
        vectordb = Chroma(collection_name="langchain",
                          embedding_function=self.embedding,
                          persist_directory="./db")
        return vectordb


def main():
    query = "gmshは二次元のmeshジェネレーターですか？ 日本語で回答してください"
    bookshelf = BookShelf(input_list)
    vectordb = bookshelf.vectordb

    qa = RetrievalQA.from_llm(llm=OpenAI(),
                              retriever=vectordb.as_retriever())
    result = qa.run(query)
    print(result)


if __name__ == "__main__":
    main()
