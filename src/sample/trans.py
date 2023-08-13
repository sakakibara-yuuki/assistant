#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
import os
import deepl

auth_key = os.getenv('DEEPL_API_KEY')
translator = deepl.Translator(auth_key)

with open('gmsh/ch1/2.txt', 'r') as f:
    result = translator.translate_text(f.read(), target_lang='JA')
print(result.text)
