#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
from bs4 import BeautifulSoup


with open('gmsh.html', 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')

print(soup.text)
