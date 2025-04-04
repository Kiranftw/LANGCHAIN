from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os
from IPython import display

class Models:
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.__API = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.__API)
        for model in genai.list_models():
            pass
        self.__LLM: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash"
        )
        R = self.__LLM.invoke("What is the capital of France?")
        
if __name__ == "__main__":
    Models()