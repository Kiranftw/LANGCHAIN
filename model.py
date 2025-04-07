from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from functools import wraps
import os

class Model():
    def __init__(self):
        load_dotenv(find_dotenv())
        if not load_dotenv(find_dotenv()):
            print("Warning: .env file not found. Environment variables may not be loaded.")
        
        API = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=API)
        for model in genai.list_models():
            if model.name == "models/gemini-2.0-flash":
                self.LLM = ChatGoogleGenerativeAI(
                    model= model.name
                )
        self.PROJECTID = "langchain-42a07"
        """
        Connection between the Python and google firebase is Complicated will do it later.
        """
    
    def ExceptionHandelling(func):
        @wraps(func)
        def Wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as E:
                print(f"Error Occured {E}" )
                return None
        return Wrapper
    
    @ExceptionHandelling        
    def message(self):
        messages = [
            SystemMessage("You are an Expert in Social Media Content Strategy"),
            HumanMessage("Give me a summary on how to get engagement in social media")
        ]
        result = self.LLM.invoke(messages)
    
    @ExceptionHandelling
    def chatbot(self):
        self.History = []
        sysmessage = SystemMessage("You are an Helpfull AI Assistant")
        self.History.append(sysmessage)
        while True:
            Query = input("You: ")
            if Query.lower() == "exit":
                break
            self.History.append(HumanMessage(content = Query))
            
            result = self.LLM.invoke(self.History)
            response = result.content
            self.History.append(AIMessage(content = response))
            print(f"AI; {response}")
            
    @ExceptionHandelling
    def promptTemplate(self):
        template = """
        Dump total knowledge on {topic} no bais and I want you to give me your opinion on 
        this {skill}.
        """
        
        messages = [
            ("system", "You are a comedian who tell jokes on pirticular god {topic}"),
            ("human", "tell me {count} jokes")
        ]
        self.promptTemplate = ChatPromptTemplate.from_messages(messages)
        prompt = self.promptTemplate.format_messages(topic="jesus", count="3")
    
    def chains(self):
       promptTemplate = ChatPromptTemplate.from_messages(
           [
               ("system", """you are actually facts expert give me some facts
               this {animal}."""),
               ("human", """tell me about this {number} of facts.""")
           ])
       chain = promptTemplate | self.LLM | StrOutputParser()
       result = chain.invoke({"animal" : "lion", "number" : 2})
       print(result)
   
if __name__ == "__main__":
    Model().chains()
