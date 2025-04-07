from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI  
from dotenv import load_dotenv, find_dotenv
import os
from IPython.display import display, Markdown

load_dotenv(find_dotenv())
if not load_dotenv(find_dotenv()):
    print("Warning: .env file not found. Environment variables may not be loaded.")
API = os.environ.get("GOOGLE_API_KEY")
LLM = ChatGoogleGenerativeAI(
    model = "models/gemini-2.0-flash",
)

summaryTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", """you are a movie critic."""),
        ("human", """give me a summary of the movie {movie}""")
    ])

def analyzePlot(movie):
    prompt = ChatPromptTemplate.from_messages(
        [   
            ("system", """you are a movie critic."""),
            ("human", """analyze the plot and provide strength and weakness of the movie {plot}""")
        ]
    )
    chain = prompt | LLM | StrOutputParser()
    return chain.invoke({"plot": movie})


def analyzeCharacter(character):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """you are a movie critic."""),
            ("human", """analyze the character and provide strength and weakness of the character {character}""")
        ]
    )
    chain = prompt | LLM | StrOutputParser()
    return chain.invoke({"character": character})


def combineVerdict(verdict1, verdict2):
    return f"""plot analysis: {verdict1}
    character analysis: {verdict2}"""

branchChain = RunnableLambda(analyzePlot)
characterChain = RunnableLambda(analyzeCharacter)


chain = (
    summaryTemplate
    | LLM
    | StrOutputParser()
    | RunnableParallel(
        {
            "plot": branchChain,
            "character": characterChain
        }
    )
    | RunnableLambda(lambda x: combineVerdict(x["plot"], x["character"]))
)


result = chain.invoke({"movie": "The Shawshank Redemption"})
print(result)