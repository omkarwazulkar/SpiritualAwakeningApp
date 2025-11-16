import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

os.environ["OPENAI_API_KEY"] = api_key

# OpenAI Client
insecure_http_client = httpx.Client(verify=False)
openai_client = OpenAI(
    api_key=api_key,
    http_client=insecure_http_client
)

# Create Query Variations
def generateQueryVariations(question: str):
    print("Generating Variations")
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant helping improve information retrieval.
        Generate five different variations of the given user question.
        Provide each variation on a new line.
        Original question: {question}"""
    )

    messages = prompt.format_messages(question=question)

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        client=openai_client.chat.completions
    )

    response = model.invoke(messages)
    print("Variations Created")

    return response.content.strip().split("\n")