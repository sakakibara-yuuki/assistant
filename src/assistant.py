from urllib.parse import urlparse
import pathlib
import argparse
from langchain.document_loaders import (
    TextLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader,
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
)
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from bs4 import BeautifulSoup
import click
import logging
import esprima
import pprint

# -[x]: input list
# -[x]: click
# -[ ]: UI
# -[ ]: server - clienet
# -[ ]: mime proceding
# -[ ]: rich
# -[ ]: pydantic
# -[ ]: API


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookShelf:
    def __init__(self, input_list:list = None):

        self.embedding = OpenAIEmbeddings()

        if input_list is not None:
            docs = self.create_documents(input_list)
            self.vectordb = self.create_db(docs)
        else:
            self.vectordb = self.load_db()

    def create_documents(self, input_list):
        loaders = []
        for uri in input_list:
            p = pathlib.Path(uri)

            if not p.exists():
                o = urlparse(uri)
                if o.scheme not in ("http", "https"):
                    raise Exception(f"{uri} cannot be found, and this is not URL")
                    continue
                loader = WebBaseLoader(uri)
                # loader = RecursiveUrlLoader(url=uri, max_depth=1, extractor=lambda x: BeautifulSoup(x, "html.parser").text)
                loaders.append(loader)
                continue

            suffix = p.suffix
            if suffix == ".txt":
                loader = TextLoader(uri)
            elif suffix == ".html":
                # loader = BSHTMLLoader(uri)
                loader = UnstructuredHTMLLoader(uri)
            # elif suffix == ".md":
            #     loader = UnstructuredMarkdownLoader(uri)
            # elif suffix == ".pdf":
            #     loader = PyPDFLoader(uri)
            # elif suffix == ".pptx":
            #     loader = UnstructuredPowerPointLoader(uri)
            # elif suffix == ".py":
            #     loader = GenericLoader.from_filesystem(
            #         uri,
            #         glob="*",
            #         suffixes=[".py"],
            #         parser=LanguageParser(language=Language.PYTHON, parser_threshold=400),
            #     )
            else:
                continue

            loaders.append(loader)

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        return docs

    def create_db(self, docs):
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)

        ### What do you use VECTOR STORE ???? ########
        vectordb = Chroma.from_documents(
            documents=documents, embedding=self.embedding, persist_directory="./db"
        )
        return vectordb

    def load_db(self):
        vectordb = Chroma(
            collection_name="langchain",
            embedding_function=self.embedding,
            persist_directory="./db",
        )
        return vectordb


@click.command()
@click.option(
    "-p",
    "--prompt",
    default="./prompt/initial",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
)
@click.option(
    "-a",
    "--answer",
    default="./answer",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, readable=False
    ),
)
@click.option(
    "-r",
    "--reference",
    default="./reference",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=False, readable=True
    ),
)
@click.option(
    "-l",
    "--link",
    default="./links",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
)
def cli(prompt, answer, reference, link):
    prompt_path = pathlib.Path(prompt)
    answer_path = pathlib.Path(answer) / prompt_path.name
    reference_path = pathlib.Path(reference)
    input_list = [str(p) for p in reference_path.glob("**/*") if (not p.is_dir()) and (p.suffix == ".py")]
    with open(link, 'r') as f:
        lines = f.readlines()
    for line in lines:
        input_list.append(line.rstrip())
    main(prompt_path, answer_path, reference_path, input_list)

def main(prompt_path, answer_path, reference_path, input_list):
    # bookshelf = BookShelf(input_list=input_list)
    bookshelf = BookShelf()
    vectordb = bookshelf.vectordb

    # LLM ラッパーの初期化
    # llm = ChatOpenAI(model_name="gpt-4", temperature=1.0, max_tokens=500)
    # chat modelだとうまくいかない
    llm = OpenAI()

    qa = RetrievalQA.from_llm(llm=llm, retriever=vectordb.as_retriever())

    with open(prompt_path, "r") as f:
        query = f.read()
    answer = qa.run(query)

    with open(answer_path, "w") as f:
        f.write(answer)
    print(answer)


if __name__ == "__main__":
    cli()
