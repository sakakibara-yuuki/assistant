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
    UnstructuredEmailLoader,
    OutlookMessageLoader,
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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from bs4 import BeautifulSoup
import click
import logging
import esprima
import pprint
from rich import print
from rich.prompt import Prompt

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
    def __init__(self, reference:str=None):

        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

        if pathlib.Path('db').exists():
            self.vectordb = self.load_db()
        else:
            if reference is None:
                raise FileNotFoundError("db is not found and reference is None")
            input_list = self.load_input_list(reference)
            docs = self.create_documents(input_list)
            self.vectordb = self.create_db(docs)


    def load_input_list(self, reference):
        """ input from reference"""
        reference_path = pathlib.Path(reference)
        input_list = [str(p) for p in reference_path.glob("**/*") if not p.is_dir()]

        """ input from links"""
        links_path = reference_path / 'links'
        if links_path.exists():
            with open(links_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                input_list.append(line.rstrip())
        return input_list


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
            elif suffix == ".md":
                loader = UnstructuredMarkdownLoader(uri)
            elif suffix == ".pdf":
                loader = PyPDFLoader(uri)
            elif suffix == ".pptx":
                loader = UnstructuredPowerPointLoader(uri)
            elif suffix == ".py":
                loader = GenericLoader.from_filesystem(
                    uri,
                    glob="*",
                    suffixes=[".py"],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=400),
                )
            elif suffix == ".eml":
                loader = UnstructuredEmailLoader(uri)
            elif suffix == ".msg":
                loader = OutlookMessageLoader(uri)
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
    "-r",
    "--reference",
    default="./reference",
    show_default=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=False, readable=True
    ),
)
def cli(reference):
    main(reference)

def main(reference):

    bookshelf = BookShelf(reference)
    vectordb = bookshelf.vectordb

    llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory)

    while True:
        query = Prompt.ask("[cyan]you [/cyan]")
        if query == "see you.":
            print("[red]A   :[/red][italic red]bye.[/italic red]")
            break
        result = qa({"question": query})
        print("[red]A   :[/red]", end="")
        print(result["answer"])


if __name__ == "__main__":
    cli()
