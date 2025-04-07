from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv
from functools import wraps
from IPython.display import Markdown, display
import google.generativeai as genai
import os
import time

def PMarkdown(text: str):
    display(Markdown(text))
    
class CHATBOT(object):
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.__API = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key = self.__API)
        self._LLM = None
        for model in genai.list_models():
            if model.name == "models/gemma-3-27b-it":
                self._LLM: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
                    model = model.name,
                    temperature = 0.5
                )
                break
        if self._LLM == None:
            raise ValueError("MODEL NOT FOUND")
        self.memory = ConversationSummaryBufferMemory(
            llm = self._LLM,
            max_token_limit=10000,
            return_messages=True
        )
        self.conversation = ConversationChain(
            llm = self._LLM,
            memory = self.memory,
            verbose = True
        )
    
    @staticmethod
    def ExceptionHandelling(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as E:
                print(f"Exception Occured {E}")
        return wrapper
    
    def chat(self, Query: str) -> str:
        response = self.conversation.predict(input=Query)
        return response
    
    def chatstream(self, Query: str) -> str:
        print("\n[Bot]: ", end="", flush=True)
        response = ""
        buffer = ""
        LASTFLUSH = time.time()
        messages = self.memory.chat_memory.messages.copy()
        messages.append(HumanMessage(content = Query))
        for chunk in self._LLM.stream(messages):
            print(chunk.content, end="", flush=True)
            buffer += chunk.content
            response += chunk.content

            if time.time() - LASTFLUSH > 0.10:
                print(chunk.content, end="", flush=True)
                buffer = ""
                LASTFLUSH = time.time()
                
        print(buffer)
        self.memory.chat_memory.add_user_message(Query)
        self.memory.chat_memory.add_ai_message(response)
        PMarkdown(response)

    
if __name__ == "__main__":
    bot = CHATBOT()
    while True:
        prompt = input("\n[You]: ")
        if prompt.lower() in ("exit", "quit"):
            break
        bot.chatstream(prompt)

