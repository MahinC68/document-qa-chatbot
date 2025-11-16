import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Load variables from .env into environment
load_dotenv()

# Read Azure OpenAI settings from environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """Return an Azure OpenAI Embedding model configured for LangChain."""
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT]):
        raise RuntimeError("Azure OpenAI embedding configuration missing in .env")

    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def get_chat_model() -> AzureChatOpenAI:
    """Return an Azure OpenAI Chat (LLM) model configured for LangChain."""
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT]):
        raise RuntimeError("Azure OpenAI chat configuration missing in .env")

    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.2,  # Low temp = more deterministic answers for Q&A
    )
