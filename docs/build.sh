#! /bin/bash
#
# build.sh
# Copyright (C) 2023 sakakibara <sakakibara@dyana>
#
# Distributed under terms of the MIT license.
#


make html
mv ./html/* ./
rm -rf html
rm -rf doctrees
