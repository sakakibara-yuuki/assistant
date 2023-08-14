import pathlib
import argparse
from langchain.document_loaders import (
    TextLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader,
    WebBaseLoader,
    PyPDFLoader,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


input_list = [
    "https://onelab.info/",
    "./reference/gmsh.html",
    "./reference/gmsh_paper_preprint.pdf",
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
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt')
    parser.add_argument('-r', '--reference')
    parser.add_argument('-a', '--answer')
    args = parser.parse_args()

    if args.prompt is None:
        raise FileNotFoundError
    prompt_path = pathlib.Path(args.prompt)

    if args.answer is None:
        answer_path = pathlib.Path('./answer') / prompt_path.name
    else:
        answer_path = pathlib.Path(args.answer)

    if args.reference is None:
        reference_path = pathlib.Path('./gmsh/gmsh.html')
    else:
        reference_path = pathlib.Path(args.reference)

    bookshelf = BookShelf(input_list)
    vectordb = bookshelf.vectordb

    # LLM ラッパーの初期化
    llm = ChatOpenAI(model_name="gpt-4",
                     temperature=0.9,
                     max_tokens=500)

    qa = RetrievalQA.from_llm(llm=llm,
                              retriever=vectordb.as_retriever())

    with open(prompt_path, 'r') as f:
        query = f.read()
    answer = qa.run(query)

    with open(answer_path, 'w') as f:
        f.write(answer)

    print(answer)
