from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain_openai import OpenAI

# Use LLM interface instead of ChatOpenAI
from langchain_openai import OpenAI  # Use this for raw LLM prompt (not chat format)

llm = OpenAI(
    temperature=0.8,
    max_tokens=256,
    openai_api_key="EMPTY",
    openai_api_base="https://pllama-7b-instruct-ws-4b-8000-e7659e.ml.iit-ropar.truefoundry.cloud/v1",
    model="xianjun-pllama-7b-instruct"
)

# Define the plain text prompt (not chat-style)
template = PromptTemplate(
    input_variables=["crop", "disease"],
    template="""
You are an AI assistant that helps Indian farmers.

A farmer's {crop} crop is affected by {disease}.
List clear, simple, and effective precautions or treatments that the farmer can take to protect the crop. 
Explain in simple terms suitable for rural farmers.
"""
)

chain = LLMChain(llm=llm, prompt=template)

def get_precautions_for_disease(crop: str, disease: str) -> str:
    response = chain.invoke({"crop": crop, "disease": disease})
    return response["text"]
