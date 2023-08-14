#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
import os
import chromadb
from chromadb.utils import embedding_functions


client = chromadb.PersistentClient(path='./db')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-ada-002",
)

# collection = client.create_collection("sample_collection", embedding_function=openai_ef)

# collection.add(
#         documents=["This is a document1", "This is document2"],
#         metadatas=[{"source": "notion"}, {"source": "google-docs"}],
#         ids=["doc1", "doc2"],
# )

collection = client.get_collection(name="sample_collection", embedding_function=openai_ef)

results = collection.query(
    query_texts=["This is a query document"],
    # query_texts=["Are you human?"],
    n_results=2,
)

print(results)
