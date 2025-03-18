import os
from griptape.chunkers import TextChunker
from griptape.drivers.embedding.ollama import OllamaEmbeddingDriver
from griptape.drivers.vector.pgvector import PgVectorVectorStoreDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.rerank.local import LocalRerankDriver
from griptape.rules import Rule, Ruleset
from griptape.loaders import WebLoader
from griptape.structures import Agent
from griptape.tools import VectorStoreTool
from griptape.utils import Chat
from griptape.tools import VectorStoreTool, RagTool
from griptape.engines import RagEngine
from griptape.engines.rag.modules import (
    PromptResponseRagModule,
    TextChunksRerankRagModule,
    VectorStoreRetrievalRagModule,
)
from griptape.engines.rag.stages import ResponseRagStage, RetrievalRagStage

import time
import concurrent.futures

# Get environment variables
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Set pgvector variables
db_user = "ian"
db_pass = DB_PASSWORD
db_host = "localhost"
db_port = "5432"
db_name = "postgres"


def process_item(item):
    # process items. item is a tuple with the URI and the namespace
    print(f"Processing tuple: {item}")
    print(f"Loading URI: {item[0]}")
    webpage_artifact = WebLoader().load(item[0])
    print(f"Chunking web page content: {item[0]}")
    chunks = TextChunker(max_tokens=250).chunk(webpage_artifact)
    print(f"Upserting chunks from: {item[0]} in namespace: {item[1]}")
    vector_store_driver.upsert_text_artifacts({item[1]: chunks})


# define function to process a list in threads
def process_list_in_threads(items):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Retrieve the result to catch any exceptions
            except Exception as e:
                print(f"An error occurred: {e}")


# Create the connetion string
db_connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Create the PgVectorVectorStoreDriver
vector_store_driver = PgVectorVectorStoreDriver(
    connection_string=db_connection_string,
    embedding_driver=OllamaEmbeddingDriver(model="all-minilm"),
    table_name="vectors",
)

# Install required Postgres extensions and create database schema
vector_store_driver.setup()

urls_to_upsert = [  # list your PDF documents here. Each tuple should contain a PDF file path and a namespace.
    ("https://griptape.ai/cloud", "Griptape Product Information"),
    ("https://griptape.ai/ai-framework", "Griptape Product Information"),
    ("https://www.griptape.ai/griptape-local", "Griptape Product Information"),
]

if True:
    print("Upserting documents in threads, one thread per document")
    t1 = time.perf_counter()  # Start the timer
    process_list_in_threads(urls_to_upsert)
    t2 = time.perf_counter()  # Stop the timer
    print("Completed with ", t2 - t1, " seconds elapsed")  # Print the time elapsed

local_rerank_driver = LocalRerankDriver(
    embedding_driver=OllamaEmbeddingDriver(model="all-minilm")
)

# Create the tool
rag_tool = RagTool(
    description="Contains information related to Griptape Products",
    off_prompt=False,
    rag_engine=RagEngine(
        retrieval_stage=RetrievalRagStage(
            retrieval_modules=[
                VectorStoreRetrievalRagModule(
                    vector_store_driver=vector_store_driver,
                    query_params={
                        "namespace": "Griptape Product Information",
                        "top_n": 20,
                    },
                )
            ],
            rerank_module=TextChunksRerankRagModule(rerank_driver=local_rerank_driver),
        ),
        response_stage=ResponseRagStage(
            response_modules=[
                PromptResponseRagModule(
                    prompt_driver=OllamaPromptDriver(model="mistral:latest")
                )
            ]
        ),
    ),
)


vector_store_tool = VectorStoreTool(
    vector_store_driver=vector_store_driver,
    description="This DB has information about Griptape Products",  # update with your topics
)

# Add code to create rules here
rag_ruleset = [
    Ruleset(
        name="RAG Ruleset",
        rules=[
            Rule(
                "Always provide citations for information that you retrieve via the RagTool. Be sure to do this every time you access information via a RagTool, even if that is only validation purposes. Never omit citations."
            ),
            Rule(
                "Only use a RagTool to find information. Do not provide information from other sources. Always validate your existing knowledge against information from the RagTool"
            ),
            Rule(
                "You are product marketing manager for Griptape, an application platform designed to accelerate the development of LLM-Powered Applications."
            ),
        ],
    )
]

prompt_driver = OllamaPromptDriver(model="mistral:latest")

# Create the agent
agent = Agent(prompt_driver=prompt_driver, tools=[rag_tool], rulesets=rag_ruleset)

# Run the agent
Chat(agent).start()
