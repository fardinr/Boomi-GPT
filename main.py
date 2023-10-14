from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import textwrap

os.environ['OPENAI_API_KEY'] = "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"
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name = model_name)
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)
embeddings = model_norm
db = Chroma(persist_directory = "C:/fardin/Icrux_Systems/OpenAI Project/New DB",
            embedding_function = embeddings)

retriever = db.as_retriever(search_kwargs={"k": 5})

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text
def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    

    chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  verbose=True,
                                  return_source_documents=True)
    matching_doc = db.similarity_search(query, k=4)
    responce = chain(query)
    print (process_llm_response(responce))
    # print()
    # print()
    # print()
    # print()

    # chain = load_qa_chain(llm=llm,chain_type="stuff",verbose=True)

    # print(chain.run({"input_documents": matching_doc, "question": query},))