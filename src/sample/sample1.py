#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
############
# テキストローダーをインポート
from langchain.document_loaders import TextLoader

# テキストローダーの初期化
loader = TextLoader('./gmsh/ch1/0.txt')

# ドキュメントの読みこみ
documents = loader.load()
############
# チャンクサイズの制限を下回るまで再帰的に分割するテキストスプリッターのインポート
from langchain.text_splitter import RecursiveCharacterTextSplitter

# テキストスプリッターの初期化
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)

# テキストをチャンクに分割
texts = text_splitter.split_documents(documents)
##############
# 使用するエンベッディングをインポート
from langchain.embeddings import OpenAIEmbeddings

# エンベッディングの初期化
embeddings = OpenAIEmbeddings()
##############
# vectorstore をインポート (ここでは Chroma を使用)
from langchain.vectorstores import Chroma

# ベクターストアにドキュメントとエンベッディングを格納
db = Chroma.from_documents(texts, embeddings)

##############
retriever = db.as_retriever()
##############

# OpenAI を使うためのインポート
from langchain.llms import OpenAI

# LLM ラッパーの初期化
llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=500)

# 質問と回答の取得に使用するチェーンをインポート
from langchain.chains import RetrievalQA

# チェーンを作り、それを使って質問に答える
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "gmshはいくつの要素から構成されている？"
answer = qa.run(query)
print(answer)
