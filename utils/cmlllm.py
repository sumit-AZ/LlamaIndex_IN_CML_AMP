import os
import asyncio
import logging
import sys
import torch
import time
import atexit
import subprocess

from dotenv import load_dotenv
from mistral_ai.core.node_parser import SimpleNodeParser
from mistral_ai.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
)
from mistral_ai.readers.file import UnstructuredReader, PDFReader
from mistral_ai.embeddings.huggingface import HuggingFaceEmbedding
from mistral_ai.vector_stores.milvus import MilvusVectorStore
from mistral_ai.models import MistralModel
from mistral_ai.core.callbacks import DebugHandler, CallbackManager
from mistral_ai.core.chat_engine.types import ChatMode
from mistral_ai.core.postprocessor import SentenceEmbeddingOptimizer
from utils.duplicate_preprocessing import DuplicateRemoverNodePostprocessor
from utils.common import supported_llm_models, supported_embed_models
import utils.vectordb as vectordb
import streamlit as st

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Constants
QUESTIONS_FOLDER = "questions"

# Exit handler
def exit_handler():
    print("Application is exiting!")
    vectordb.stop_vector_db()

atexit.register(exit_handler)

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Debug and Callback setup
debug_handler = DebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[debug_handler])

# Reset questions folder
if os.path.exists(QUESTIONS_FOLDER):
    for file in os.listdir(QUESTIONS_FOLDER):
        file_path = os.path.join(QUESTIONS_FOLDER, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    os.rmdir(QUESTIONS_FOLDER)

os.makedirs(QUESTIONS_FOLDER, exist_ok=True)

# Reset and start vector DB
milvus_start = vectordb.reset_vector_db()
print(f"Milvus started: {milvus_start}")

# Chat engine map
chat_engine_map = {}

# Active collection map
active_collection_available = {"Default": False}

# Streamlit caching decorator
@st.cache_resource
class CMLLLM:
    MODELS_PATH = "./models"
    EMBED_PATH = "./embed_models"
    questions_folder = QUESTIONS_FOLDER

    def __init__(
        self,
        model_name="mistralai/Mistral-7B",
        embed_model_name="thenlper/gte-large",
        temperature=0.0,
        max_new_tokens=1024,
        context_window=3900,
        gpu_layers=20,
        dim=1024,
        collection_name="Default",
        memory_token_limit=3900,
        sentense_embedding_percentile_cutoff=0.8,
        similarity_top_k=2,
        progress_bar=None,
    ):
        # Set defaults if models are not provided
        model_name = model_name or "mistralai/Mistral-7B"
        embed_model_name = embed_model_name or "thenlper/gte-large"
        self.active_model_name = model_name
        self.active_embed_model_name = embed_model_name
        n_gpu_layers = gpu_layers if torch.cuda.is_available() else 0

        print(f"Using {'GPU' if n_gpu_layers else 'CPU'}.")

        self.node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=128)

        self.set_global_settings(
            model_name=model_name,
            embed_model_path=embed_model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=n_gpu_layers,
            node_parser=self.node_parser,
            progress_bar=progress_bar,
        )
        self.dim = dim
        self.similarity_top_k = similarity_top_k
        self.sentense_embedding_percentile_cutoff = sentense_embedding_percentile_cutoff
        self.memory_token_limit = memory_token_limit

    def set_global_settings(
        self,
        model_name,
        embed_model_path,
        temperature,
        max_new_tokens,
        context_window,
        n_gpu_layers,
        node_parser,
        progress_bar=None,
    ):
        print(
            f"Setting global settings: model={model_name}, embedding={embed_model_path}"
        )
        self.active_model_name = model_name
        self.active_embed_model_name = embed_model_path
        model_path = self.get_model_path(model_name)
        print(f"Model path: {model_path}")

        Settings.llm = MistralModel(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            gpu_layers=n_gpu_layers,
            verbose=True,
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_path,
            cache_folder=self.EMBED_PATH,
        )
        Settings.node_parser = node_parser

    def get_model_path(self, model_name):
        filename = supported_llm_models.get(model_name)
        if not filename:
            raise ValueError(f"Unsupported model: {model_name}")
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            resume_download=True,
            cache_dir=self.MODELS_PATH,
            local_files_only=True,
            token=hf_token,
        )
        return model_path

def infer2(msg, history, collection_name):
    query_text = msg.strip()
    print(f"Query: {query_text}, Collection: {collection_name}")

    if not query_text:
        return "Please ask some questions."

    if collection_name not in chat_engine_map:
        return f"Chat engine not created for collection {collection_name}."

    chat_engine = chat_engine_map[collection_name]

    try:
        streaming_response = chat_engine.stream_chat(query_text)
        for token in streaming_response.response_gen:
            yield token
    except Exception as e:
        logging.error(f"Error: {e}")
        yield f"Failed with exception: {e}"
