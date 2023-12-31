#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
import re

import click
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from rich import print
from rich.prompt import Prompt

from bookshelf import BookShelf


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    ctx.obj["bookshelf"] = BookShelf()


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
    bookshelf = ctx.obj["bookshelf"]
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
@click.option("-i", "--interactive", default=False, show_default=True, type=bool)
def qa(ctx, prompt, interactive):
    bookshelf = ctx.obj["bookshelf"]
    if interactive is True:
        prompt = Prompt.ask("[cyan]you [/cyan]")
    qa_mode(bookshelf.vectordb, prompt)


@cli.command()
@click.pass_context
def chat(ctx):
    bookshelf = ctx.obj["bookshelf"]
    chat_mode(bookshelf.vectordb)


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
def summary(ctx, prompt):
    bookshelf = ctx.obj["bookshelf"]
    if prompt is not None:
        with open(prompt, 'r') as f:
            prompt = f.read()
    else:
        prompt = Prompt.ask("[cyan]you [/cyan]")
    summary_mode(bookshelf.vectordb, prompt)


def summary_mode(vectordb, prompt):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)
    chain = load_summarize_chain(llm, chain_type="stuff")
    search = vectordb.similarity_search(" ")
    answer = chain.run(input_documents=search, question=prompt)
    print(answer)


def chat_mode(vectordb):
    llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm, vectordb.as_retriever(), memory=memory, max_tokens_limit=8000
    )

    while True:
        query = Prompt.ask("[cyan]you [/cyan]")
        if re.match("(Bye|bye|BYE).*", query) is not None:
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
