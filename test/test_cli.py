#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.

import pytest
import copy
from assistant.bookshelf import BookShelf


def test_create_bookshelf():
    bookshelf = BookShelf()
    assert isinstance(bookshelf, BookShelf)


def test_show_bookshelf_list():
    list_json = bookshelf.list
    assert sample_list == list_json


def test_create_book():
    bookshelf = BookShelf()
    be_bookshelf = copy(bookshelf)
    bookshelf.create_book(reference: path)
    be_list = bookshelf.list
    af_list = bookshelf.list
    assert be_list != af_list


def test_read_book():
    bookshelf = BookShelf()
    book = bookshelf.read(book_info)
    assert book_is_readable


def test_update_book():
    bookshelf = BookShelf()
    bookshelf.update(reference.yaml)


def test_delete_book():
    bookshelf = BookShelf()
    bookshelf.delete(book)

