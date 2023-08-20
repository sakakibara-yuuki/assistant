#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
import click
from assistant import BookShelf
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


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    ctx.obj['bookshelf'] = BookShelf()


@cli.command()
@click.pass_context
@click.option(
    "-u",
    "--update",
    default=None,
    show_default=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
)
def bookshelf(ctx, update):
    bookshelf = ctx.obj['bookshelf']
    if update is not None:
        bookshelf.update(update)


@cli.command()
@click.pass_context
@click.option(
    "-p",
    "--prompt",
    default=None,
    show_default=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, writable=False, readable=True
    ),
)
@click.option(
    "-i",
    "--interactive",
    default=False,
    show_default=True,
    type=bool
)
def qa(ctx, prompt, interactive):
    bookshelf = ctx.obj['bookshelf']
    if interactive is True:
        prompt = Prompt.ask("[cyan]you [/cyan]")
    qa_mode(bookshelf.vectordb, prompt)


@cli.command()
@click.pass_context
def chat(ctx):
    bookshelf = ctx.obj['bookshelf']
    chat_mode(bookshelf.vectordb)


def chat_mode(vectordb):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm, vectordb.as_retriever(), memory=memory, max_tokens_limit=8000
    )

    while True:
        query = Prompt.ask("[cyan]you [/cyan]")
        if re.match('(Bye|bye|BYE).*', query) is not None:
            print("[red]A   :[/red][italic red]bye![/italic red]")
            return
        result = qa({"question": query})
        print("[red]A   :[/red]", end="")
        print(result["answer"])


def qa_mode(vectordb, prompt):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)
    qa = RetrievalQA.from_llm(llm=llm, retriever=vectordb.as_retriever())
    answer = qa.run(prompt)
    print(answer)


if __name__ == "__main__":
    cli()
