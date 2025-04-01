from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

class MemoryChatbot:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.__API = os.getenv("GEMINIAPI")
        genai.configure(api_key=self.__API)

        self.__MODEL = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")

        # Function to retrieve or create a session history
        def get_session_history(session_id: str) -> ChatMessageHistory:
            if session_id not in self.sessions:
                self.sessions[session_id] = ChatMessageHistory()
            return self.sessions[session_id]  

        # Dictionary to store chat histories for different sessions
        self.sessions = {}

        # Use RunnableWithMessageHistory with get_session_history function
        self.Conversation = RunnableWithMessageHistory(
            self.__MODEL, 
            get_session_history=get_session_history
        )

    def start_conversation(self, user_input: str, session_id: str = "default_session") -> str:
        # Append user input to memory
        self.sessions[session_id].add_message(HumanMessage(content=user_input))

        # Get response from model, passing session_id in the config
        response = self.Conversation.invoke(
            input=user_input,
            config={"configurable": {"session_id": session_id}}
        )

        # Store bot's response
        self.sessions[session_id].add_message(AIMessage(content=response))

        return response

if __name__ == "__main__":
    chatbot = MemoryChatbot()
    
    print("Chatbot is running! Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nYou: ")  
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        response = chatbot.start_conversation(user_input)  
        print(f"Bot: {response}")
