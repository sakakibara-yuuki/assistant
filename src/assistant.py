import pathlib
import argparse
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


class Reference:
    def __init__(self, file_path):
        documents = self.documents_loader(file_path)
        texts = self.make_text(documents)
        self._retriever = self.make_embedding(texts)

    @property
    def retriever(self):
        return self._retriever


    def documents_loader(self, file_path):
        # loader = TextLoader(file_path)
        loader = UnstructuredHTMLLoader('./gmsh/gmsh.html')
        documents = loader.load()

        return documents

    def make_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    def make_embedding(self, texts):
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        return retriever


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt')
    parser.add_argument('-r', '--reference')
    parser.add_argument('-a', '--answer')
    args = parser.parse_args()

    if args.prompt is None:
        raise FileNotFoundError
    prompt_path = pathlib.Path(args.prompt)

    if args.answer is None:
        answer_path = pathlib.Path('./answer') / prompt_path.name
    else:
        answer_path = pathlib.Path(args.answer)

    if args.reference is None:
        reference_path = pathlib.Path('./gmsh/gmsh.html')
    else:
        reference_path = pathlib.Path(args.reference)
    reference = Reference(reference_path)
    retriever = reference.retriever

    # LLM ラッパーの初期化
    # llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=500)
    llm = ChatOpenAI(model_name="gpt-4",
                     temperature=0.9,
                     max_tokens=500)

    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever)

    with open(prompt_path, 'r') as f:
        query = f.read()
    answer = qa.run(query)

    with open(answer_path, 'w') as f:
        f.write(answer)

    print(answer)
