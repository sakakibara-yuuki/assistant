#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
from bs4 import BeautifulSoup
import deepl
import os

translator = deepl.Translator(auth_key=os.getenv('DEEPL_API_KEY'))

with open("./reference/gmsh.html") as fp:
    soup = BeautifulSoup(fp, 'html.parser')

text = soup.get_text()
result = translator.translate_text(text, target_lang="JA")
translated_text = result.text
print(translated_text)
