from typing import List, Sequence

import chromadb
from llama_index.core import StorageContext, Document, Settings, get_response_synthesizer
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, MarkdownNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_wrapper import LlamaWrapper, Inferix, OpenAIModelInterface

Settings.embed_model = TextEmbeddingsInference(model_name='some-model', base_url='http://localhost:7000')
Settings.llm = LlamaWrapper(ai_obj=OpenAIModelInterface("Llama-3-8B-Instruct", 1 << 28, Inferix(
    "base-url",
    "api-key"
)))


def store_embeddings(texts: List[str]) -> Sequence[BaseNode]:
    docs = [Document(text=text) for text in texts]
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=100, chunk_overlap=50),
        MarkdownNodeParser(),
        HierarchicalNodeParser.from_defaults(chunk_sizes=[4096, 2048, 512]),
    ])
    ingested_nodes = pipeline.run(documents=docs)
    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
    automerging_index = VectorStoreIndex(
        ingested_nodes,
        storage_context=storage_context,
        show_progress=True,
        insert_batch_size=1 << 11,
    )
    automerging_index.storage_context.persist(persist_dir='./storage/storage_context')
    return ingested_nodes


def query(question):
    vs = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vs)
    retriever = VectorIndexRetriever(index, similarity_top_k=10)
    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=get_response_synthesizer()
    )
    return query_engine.query(question)


def get_vector_store():
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection("test_embeddings")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


store_embeddings(['1 is 2', '3 is 4', '5 is 6'])
print(query("What is 1 + 3?"))
