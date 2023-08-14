#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
from .input import BookShelf

book = {name: "gmsh-html",
        uri: "http://index.html",
        doc_type: html}


def test_use_case:
    input_list = ["https://onelab.info/",
                  "../reference/gmsh.html",
                  "../reference/gmsh_paper_preprint.pdf"]
    bookshelf = BookShelf(input_list)

def test_bookshelf_add():
    input_list = ["https://onelab.info/",
                  "../reference/gmsh.html"]
    bookshelf = BookShelf(input_list)
    bookshelf.add("../reference/gmsh_paper_preprint.pdf")
    input_list = ["https://onelab.info/",
                  "../reference/gmsh.html",
                  "../reference/gmsh_paper_preprint.pdf"]
    assert bookshelf.ls == input_list

def test_bookshelf_delete():
    bookshelf = BookShelf(input_list)
    bookshelf.delete(book_name)
    assert bookshelf.ls == []

def test_bookshelf_update():
    bookshelf = BookShelf(input_list)
    bookshelf.delete(book_name)
    assert bookshelf.ls == []

def test_bookshelf_read():
    bookshelf = BookShelf(input_list)
    text = bookshelf.read(book_name)
    assert isinstance(text, str)
