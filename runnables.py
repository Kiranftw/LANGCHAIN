from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os
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
        self.corpus = ""

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
        print("\n[Bot]: ", end="", flush=True)  # Print the initial prompt
        response = ""  # Initialize the response string
        messages = self.memory.chat_memory.messages.copy()
        messages.append(HumanMessage(content=Query))
        for chunk in self.model.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)  # Print each chunk immediately
                response += chunk.content  # Accumulate the response
        
        print()

        self.memory.chat_memory.add_user_message(Query)
        self.memory.chat_memory.add_ai_message(response)
        PMarkdown(response)
        return response
    
    @ExceptionHandelling
    def pdfchatbot(self, Document: str, Query: str) -> str:
        loader = PyMuPDFLoader(Document)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Combine all text chunks into a single string
        combined_text = " ".join([doc.page_content for doc in texts])
        
        self.memory.chat_memory.add_user_message(combined_text)
        self.memory.chat_memory.add_ai_message("Document Loaded Successfully")
        
        if Query:
            print("\n[Bot]: ", end="", flush=True)
            response = ""
            messages = self.memory.chat_memory.messages.copy()
            messages.append(HumanMessage(content=Query))
            for chunk in self.model.stream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    response += chunk.content
            print()
            self.memory.chat_memory.add_user_message(Query)
            self.memory.chat_memory.add_ai_message(response)
            PMarkdown(response)
            return response
        else:
            return "Document Loaded Successfully"

if __name__ == "__main__":
    chatbot = RunnableCHATBOT()

    while True:
        document = input("Enter the PDF document path (or 'exit' to quit): ")
        if document.lower() == "exit":
            break

        chatbot.pdfchatbot(Document=document, Query="") 

        while True:
            prompt = input("\n[You]: ")
            if prompt.lower() == "exit":
                print("Ending session and clearing memory...\n")
                chatbot.memory.clear() 
                break

            chatbot.pdfchatbot(Document=document, Query=prompt)

