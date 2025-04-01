from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.base import ConversationChain

class Memory(object):
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.__API = os.getenv("GEMINIAPI")
        genai.configure(api_key=self.__API)
        self.memory = ConversationBufferMemory()
        self.__MODEL: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model = "models/gemini-2.0-flash"
        )
        self.Conversation = ConversationChain(
            llm = self.__MODEL,
            memory = self.memory,
            verbose = True
        )
    
    def conversation(self) -> str:
        print(self.Conversation.predict(input = "Hi there!"))

if __name__ == "__main__":
    Memory().Conversation()
