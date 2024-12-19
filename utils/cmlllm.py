import os
import asyncio
import time
import torch
import logging
import sys
import subprocess
import streamlit as st
import atexit
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from mistral.models import MistralModel  # Hypothetical Mistral AI model interface
from mistral.embeddings import MistralEmbedding  # Hypothetical Mistral embedding interface
from mistral.vector_stores.milvus import MilvusVectorStore  # Hypothetical Mistral vector store
from mistral.utils.parsers import SimpleNodeParser
from mistral.core.chat_engine import ChatEngine  # Hypothetical Mistral chat engine
from mistral.core.settings import Settings  # Mistral settings module
from mistral.core.memory import ChatMemoryBuffer
from mistral.core.postprocessors import SentenceEmbeddingOptimizer, DuplicateRemover

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

QUESTIONS_FOLDER = "questions"
MODELS_PATH = "./models"
EMBED_PATH = "./embed_models"

# Register exit handler
def exit_handler():
    print("Exiting application!")
    subprocess.run(["pkill -f milvus"], shell=True)  # Example vector store cleanup

atexit.register(exit_handler)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Define global variables
chat_engine_map = {}
active_collection_available = {"Default": False}

# Delete temporary questions folder
subprocess.run([f"rm -rf {QUESTIONS_FOLDER}"], shell=True)

class MistralLLM:
    def __init__(
        self,
        model_name="TheBloke/Mistral-7B-Instruct-v0.2",
        embed_model_name="mistral/gte-large",
        temperature=0.7,
        max_new_tokens=512,
        context_window=3900,
        gpu_layers=20,
        dim=1024,
        collection_name="Default",
        memory_token_limit=4000,
        similarity_top_k=5,
    ):
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.dim = dim
        self.similarity_top_k = similarity_top_k
        self.memory_token_limit = memory_token_limit

        # Initialize Mistral model
        self.llm = MistralModel(
            model_path=self.download_model(model_name),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            gpu_layers=gpu_layers,
        )

        # Initialize embedding model
        self.embed_model = MistralEmbedding(
            model_name=self.download_embedding_model(embed_model_name),
        )

        self.vector_store = None
        self.node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=128)

    def download_model(self, model_name):
        filename = f"{model_name}.bin"
        return hf_hub_download(
            repo_id=model_name,
            filename=filename,
            resume_download=True,
            cache_dir=MODELS_PATH,
            token=hf_token,
        )

    def download_embedding_model(self, embed_model_name):
        return snapshot_download(
            repo_id=embed_model_name,
            resume_download=True,
            cache_dir=EMBED_PATH,
            token=hf_token,
        )

    def initialize_vector_store(self, collection_name):
        self.vector_store = MilvusVectorStore(
            dim=self.dim,
            collection_name=collection_name,
        )

    def set_collection_name(self, collection_name):
        if collection_name not in active_collection_available:
            active_collection_available[collection_name] = False

        if collection_name not in chat_engine_map:
            print(f"Setting up collection: {collection_name}")
            self.initialize_vector_store(collection_name)
            chat_engine = ChatEngine(
                llm=self.llm,
                vector_store=self.vector_store,
                embedding_model=self.embed_model,
                memory=ChatMemoryBuffer(token_limit=self.memory_token_limit),
                postprocessors=[
                    SentenceEmbeddingOptimizer(percentile_cutoff=0.8),
                    DuplicateRemover(),
                ],
                similarity_top_k=self.similarity_top_k,
            )
            chat_engine_map[collection_name] = chat_engine
            active_collection_available[collection_name] = True

    def ingest_documents(self, files, questions, collection_name):
        if not active_collection_available.get(collection_name):
            return f"Collection {collection_name} is not active."

        node_parser = self.node_parser
        for file in files:
            # Load and parse documents
            print(f"Ingesting file: {file}")
            nodes = node_parser.parse_file(file)

            # Add parsed nodes to the vector store
            for node in nodes:
                self.vector_store.add(node)

        return "Ingestion complete."

    def chat(self, message, collection_name):
        if collection_name not in chat_engine_map:
            return f"Collection {collection_name} not found."

        chat_engine = chat_engine_map[collection_name]
        return chat_engine.chat(message)

# Streamlit UI
st.title("Mistral AI Chatbot")
collection_name = st.text_input("Collection Name", "Default")
input_message = st.text_area("Your Message")
uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

llm = MistralLLM()

if st.button("Ingest"):
    response = llm.ingest_documents(uploaded_files, questions=5, collection_name=collection_name)
    st.write(response)

if st.button("Chat"):
    response = llm.chat(input_message, collection_name)
    st.write(response)
