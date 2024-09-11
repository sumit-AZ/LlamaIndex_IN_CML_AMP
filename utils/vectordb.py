import os
import utils.vector_db_utils as vector_db


def start_vector_db():
    return vector_db.start_milvus()


def vector_db_status():
    return vector_db.get_milvus_status()


def reset_vector_db():
    return vector_db.reset_data()


def stop_vector_db():
    return vector_db.stop_milvus()


def create_or_get_vector_db_collection(collection_name="default_collection", dim=1024):
    collection_name = collection_name
    dim = dim
    vector_db.create_milvus_collection(collection_name=collection_name, dim=dim)
    return f"collection {collection_name} created with dim {dim}"
