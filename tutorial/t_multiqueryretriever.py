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
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

# [x] 埋め込みモデルをgptに変更する
# [x] 複数クエリを扱えるようにする
# [ ] データを自動分類する
# [ ] self-queryとは何かについて調べる
# [ ] webqueryの機能をつける
# [ ] 時間加重をつける
# [ ] CLIをつくる
# [ ] UIをつくる
# [ ] textsplitter について

# [x] 複数のデータローダーを使えるかを確認。
# loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
# loader = UnstructuredHTMLLoader("../reference/gmsh.html")
web_loader = WebBaseLoader("https://onelab.info/")
html_loader = BSHTMLLoader("../reference/gmsh.html")
pdf_loader = PyPDFLoader("../reference/gmsh_paper_preprint.pdf")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embedding = OpenAIEmbeddings()
# index = VectorstoreIndexCreator(
#             vectorstore_cls=Chroma,
#             embedding=embedding,
#             text_splitter=text_splitter
#         ).from_loaders([web_loader,  pdf_loader])

query = "gmshとはなんですか？"
# result = index.query(query)
# print(result)
llm = ChatOpenAI(temperature=0)
vectordb = Chroma.from_documents(documents=text_splitter.split_documents(html_loader.load()),
                                 embedding=embedding)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

unique_docs = retriever_from_llm.get_relevant_documents(query=query)
print(type(unique_docs))
print('='*31)
for d in unique_docs:
    print(d.page_content)
    print('-'*31)
