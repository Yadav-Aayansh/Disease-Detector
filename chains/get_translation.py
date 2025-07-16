from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()


# Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3
)

translation_prompt = PromptTemplate(
    input_variables=["precautions", "language"],
    template="""
You are a helpful translation assistant.

Translate the following farming advice into {language}. Keep the meaning and simplicity for rural farmers.Not include markdown give the response in plain text.

Precautions:
{precautions}
"""
)

# Create LangChain chain
translation_chain = LLMChain(llm=gemini_llm, prompt=translation_prompt)

def translate_precautions(language: str, precautions: str) -> str:
    response = translation_chain.invoke({
        "precautions": precautions,
        "language": language
    })
    return response["text"]

