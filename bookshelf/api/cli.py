#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
"""
cliは基本的に

python assistant create bookshelf
のように英語の5文型を基本とする

python assistant create bookshelf
python assistant show bookshelf
python assistant summary --about book1
python assistant add book
python assistant chat --about book1 book2
python assistant delete book1
python assistant fill template.md --about book
--aboutの後にはbookを入れる.
"""
import click


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.pass_context
@click.argument('bookshelf')
def create(ctx, bookshelf):
    """
    python assistant create bookshelf
    """
    pass


@cli.command()
@click.pass_context
@click.argument('bookshelf')
def show(ctx, bookshelf):
    """
    python assistant show bookshelf
    """
    pass


@cli.command()
@click.pass_context
@click.option(
    "-a",
    "--about",
    default=None,
    show_default=True,
    nargs=2,
)
def summary(ctx, about):
    """
    python assistant summary --about book1
    """
    book1, book2 = about
    pass


@cli.command()
@click.pass_context
@click.argument('book')
def add(ctx, book):
    """
    python assistant add book
    """
    pass


@cli.command()
@click.pass_context
@click.option(
    "-a",
    "--about",
    default=None,
    show_default=True,
    nargs=2,
)
def chat(ctx, book1, book2):
    """
    python assistant chat --about book1 book2
    """
    book1, book2 = about
    pass


@cli.command()
@click.pass_context
@click.argument('book')
def delete(ctx, book):
    """
    python assistant delete book
    """
    pass


@cli.command()
@click.pass_context
@click.argument('template')
@click.option(
    "--about",
    default=None,
    show_default=True,
)
def fill(ctx, template, about):
    """
    python assistant fill template.md --about book
    """
    pass
