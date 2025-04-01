from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate

class Memory(object):
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.__API = os.getenv("GEMINIAPI")
        genai.configure(api_key=self.__API) 
        self.__MODEL: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash" 
        )
        
        prompt_template = PromptTemplate.from_template("User: {input}")
        
        self.memory = ConversationSummaryMemory(
            llm=self.__MODEL, 
            memory_key="history",
            human_prefix="User",
            ai_prefix="Bot",
            input_key="input"
        )
        
        self.Conversation = ConversationChain(
            llm=self.__MODEL,  
            memory=self.memory,  
            verbose=True 
        )
    
    def start_conversation(self, user_input: str) -> str:
        response = self.Conversation.predict(input=user_input)
        return response

if __name__ == "__main__":
    chatbot = Memory() 
    user_input = input()
    response = chatbot.start_conversation(user_input) 
    print(chatbot)
    print(response)