#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
import os
import deepl
# from bs4 import BeautifulSoup


# with open('ch1.html', 'r') as f:
#     soup = BeautifulSoup(f, 'html.parser')

auth_key = os.getenv('DEEPL_API_KEY')
print(auth_key)

for i in range(8):
    with open(f'gmsh/ch1/{i}.txt', 'r') as f:

        text = f.read()

        translator = deepl.Translator(auth_key)
        result = translator.translate_text(text, target_lang='JA')
        print(result.text)
        # with open(f'gmsh/ch1_ja/{i}.txt', 'w') as f:
        #     f.write(result.text)
