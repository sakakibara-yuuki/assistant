# ASSISTANT FOR YOU !
This is my assistant about anything

```py
Usage: assistant.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  bookshelf
  chat
  qa
```
Document is [HERE](https://sakakibara-yuuki.github.io/assistant/)

## 🚀 Quickstart
> **Step 1** - download and install assistant with git clone & pip

```bash
$ git clone https://github.com/sakakibara-yuuki/assistant.git
$ cd assistant
$ pip install -e .
```
<br />

> **Step 2** - set OPNEAI_API_KEY
```bash
$ export OPENAI_API_KEY = "your_openai_api_key_here!"
```
<br />

> **Step 3** - Write a list of articles you want your assistant to refer to in the reference.yaml file.
```vi
files:
  - hogehoge.txt
  - hogehoge.pdf
  - https://hoghoge/index.html
```
<br />

> **Step 4** - Give the reference.yaml file to your assistant to create a bookshelf.
```bash
$ python -m assistant bookshelf -u reference.yaml
```
<br />

> **Step 5** - Let your assistant help you!
```bash
$ python -m assistant chat
> you:
or
$ python -m assistant qa -p prompt.txt
> A  :
or
$ python -m assistant qa -i 
> you:
```
<br />
