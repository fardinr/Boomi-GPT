from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings




folder = "C:/fardin/Icrux_Systems/OpenAI Project/Data"

loder = DirectoryLoader(folder) 
documents = loder.load()

# print(documents)

test_spliter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
doc = test_spliter.split_documents(documents=documents)

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = model_norm

# print (doc)

vector_db = Chroma.from_documents(documents =doc, embedding = embeddings, persist_directory = "C:/fardin/Icrux_Systems/OpenAI Project/DB")
vector_db.persist()


# import openai
# import os

# os.environ['OPENAI_API_KEY'] = "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"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# response = openai.Embedding.create(
#     input="Your text string goes here",
#     model="text-embedding-ada-002"
# )
# embeddings = response['data'][0]['embedding']
# # print(embeddings)