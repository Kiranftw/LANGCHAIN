from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
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
                
    def message(self):
        messages = [
            SystemMessage("You are an Expert in Social Media Content Strategy"),
            HumanMessage("Give me a summary on how to get engagement in social media")
        ]
        result = self.LLM.invoke(messages)
    
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
            
        

if __name__ == "__main__":
    Model().chatbot()
