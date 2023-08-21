# ASSISTANT FOR YOU !
**A diligent assistant will assist you!**

Have you ever felt the need to follow a document or URL when using LLM ?
It's your assistant's turn!
The assistant receives a list of references and creates his own bookshelf.
He will chat with us and answer QA based on the contents of his own bookshelf!

## 📝 Requirements
- git
- python >= 3.11
- [OPENAI_API_KEY](https://platform.openai.com/)

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
> you: what is prarie dog?
> A  : prarie dog is ...
> you: bye
> A  : bye!
```
or

```bash
$ python -m assistant qa -p prompt.txt
> A  : prarie dog is ...
```

or

```bash
$ python -m assistant qa -i 
> you:  what is prarie dog?
> A  : prarie dog is ...
```

<br />

## Documents
For more detailed programme content, click [HERE](https://sakakibara-yuuki.github.io/assistant/)!!
