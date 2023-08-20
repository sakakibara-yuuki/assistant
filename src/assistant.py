import re
import yaml
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
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory
)
from pydantic import BaseModel
from bs4 import BeautifulSoup
import click
import logging
import esprima
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
    def __init__(self, reference:str):
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.input_list = []
        self.vectordb = None
        self.data = self.load_yaml(reference)

        if pathlib.Path("db").exists():
            self.vectordb = self.load_db()
            return

        files = self.data['files']
        self.input_list = self.load_reference(files)
        docs = self.create_documents(self.input_list)
        self.vectordb = self.create_db(docs)

    def load_yaml(self, reference):
        reference_path = pathlib.Path(reference)

        if not reference_path.exists():
            raise FileNotFoundError("reference is not Found")

        if reference_path.is_dir():
            raise FileNotFoundError("reference must be file not dir")

        with open(reference_path, "r") as f:
            data = yaml.safe_load(f)

        return data

    def load_reference(self, files):

        input_list = []

        for uri in files:
            """ when line is url """
            o = urlparse(uri)
            if o.scheme in ("http", "https"):
                input_list.append(uri)
                continue

            """ when line is directory path """
            p = pathlib.Path(uri)
            if p.is_dir():
                for _p in p.glob("**/*"):
                    if _p.is_dir():
                        continue
                    input_list.append(str(_p))
            else:
                input_list.append(str(p))

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
                loaders.append(loader)
                continue

            suffix = p.suffix
            if suffix == ".txt":
                loader = TextLoader(uri)
            elif suffix == ".html":
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
                    parser=LanguageParser(
                        language=Language.PYTHON, parser_threshold=400
                    ),
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        # text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)

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
