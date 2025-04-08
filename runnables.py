from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from functools import wraps
from IPython.display import Markdown, display

def PMarkdown(text: str) -> None:
    display(Markdown(text))

def ExceptionHandelling(func) -> object:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as E:
            print(f"Exception Occured {E}")
    return wrapper


class RunnableCHATBOT(object):
    load_dotenv(find_dotenv())
    @ExceptionHandelling
    def __init__(self) -> None:
        self.API = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key = self.API)
        for model in genai.list_models():
            pass
        self.model = None
        for model in genai.list_models():
            if model.name == "models/gemini-2.0-flash":
                self.model = ChatGoogleGenerativeAI(
                    model = model.name,
                    temperature = 0.5
                )
                break
        if self.model == None:
            raise ValueError("MODEL NOT FOUND")
        self.memory = ConversationSummaryBufferMemory(
            llm = self.model,
            max_token_limit=10000,
            return_messages=True
        )
        self.conversation = ConversationChain(
            llm = self.model,
            memory = self.memory,
            verbose = True
        )

    @ExceptionHandelling
    def getResponse(self, Query: str) -> str:
        response = self.MODEL.invoke(Query)
        return response.content
    
    @ExceptionHandelling
    def withoutMemory(self):
        memory = []
        sysmessage = SystemMessage("You are an Helpfull AI Assistant")
        memory.append(sysmessage)
        while True:
            Query = input("You: ")
            if Query.lower() == "exit":
                break
            memory.append(HumanMessage(content = Query))
            result = self.getResponse(memory)
            memory.append(AIMessage(content = result))
            print(f"AI: {result}")
            PMarkdown(f"AI: {result}")
    
    @ExceptionHandelling
    def withMemory(self, Query: str) -> str:
        print("\n[Bot]: ", end="", flush=True)
        response = ""
        buffer = ""
        lastflush = time.time()
        messages = self.memory.chat_memory.messages.copy()
        messages.append(HumanMessage(content = Query))
        for chunk in self.model.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                buffer += chunk.content
                response += chunk.content

                if time.time() - lastflush > 0.87:
                    buffer = ""
                    lastflush = time.time()
        print(buffer)
        self.memory.chat_memory.add_user_message(Query)
        self.memory.chat_memory.add_ai_message(response)
        PMarkdown(response)
        return response

if __name__ == "__main__":
    chatbot = RunnableCHATBOT()
    while True:
        prompt = input("\n[You]: ")
        if prompt.lower() in ("exit", "quit"):
            break
        chatbot.withMemory(prompt)