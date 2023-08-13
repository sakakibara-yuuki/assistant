#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
# ドキュメントローダーをインポート
from langchain.document_loaders import TextLoader
# ドキュメントローダーの初期化
loader = TextLoader('./gmsh/ch1/0.txt')

# インデックスの作成に用いるクラスをインポート
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# ベクターストアの作成
index: VectorStoreIndexWrapper = VectorstoreIndexCreator().from_loaders([loader])
query = "gmshはいくつの要素から構成されている？"
answer = index.query(query)

print(answer)
