from milvus import default_server
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
import socket
from contextlib import closing
import subprocess


def start_milvus():
    if check_socket("localhost", default_server.listen_port):
        return utility.get_server_version()

    # Start Milvus Vector DB
    default_server.set_base_dir("milvus-data")
    default_server.start()

    try:
        connections.connect(
            alias="default", host="localhost", port=default_server.listen_port
        )
    except Exception as e:
        default_server.stop()
        raise e

    return utility.get_server_version()


def stop_milvus():
    # Stop Milvus Vector DB
    default_server.stop()
    return "Milvus stopped"


def get_milvus_status():
    if check_socket("localhost", default_server.listen_port):
        return f"milvus is running. version = {utility.get_server_version()}"

    return f"milvus is stopped"


def reset_data():
    stop_milvus()
    print(subprocess.run(["rm -rf milvus-data"], shell=True))
    return start_milvus()


def check_socket(host, port) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            print(f"Port {port} is open")
            return True
        else:
            print(f"Port {port} is not open")
    return False


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        print(f"collection {collection_name} already exists")
        return Collection(collection_name)

        # utility.drop_collection(collection_name)

    fields = [
        FieldSchema(
            name="relativefilepath",
            dtype=DataType.VARCHAR,
            description="file path relative to root directory ",
            max_length=1000,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            description="embedding vectors",
            dim=dim,
        ),
    ]
    schema = CollectionSchema(fields=fields, description="reverse image search")
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection
