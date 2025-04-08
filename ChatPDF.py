import os
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

class CHATBOT(object):
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        genai.configure(api_key = "GOOGLE_API_KEY")
        for model in genai.list_models():
            print(model.name)


if __name__ == "__main__":
    object = CHATBOT()
        
