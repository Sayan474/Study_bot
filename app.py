import datetime
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime, UTC
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

client = MongoClient(mongo_uri)
db = client["chat"]
collection = db["users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI-powered Study Assistant.
            You help users with academic and study-related questions.
            You are allowed to:
            - Greet users politely
            - Explain who you are
            - Explain what you can do
            - Ask for clarification if needed
            - Answer academic questions

            Allowed academic topics include:
            Mathematics, Computer Science, Programming,
            Physics, Chemistry, Biology, Engineering,
            History, Geography, Economics, Exam preparation.

            Use simple plain text formatting.
            Write mathematical expressions in readable text form (e.g., sqrt(x), x^2, sin(x)).
            Do not use LaTeX delimiters like \[ \] or \( \).
            Avoid Markdown symbols like **.

            If a user asks about topics unrelated to academics
            (such as entertainment, jokes, politics, gossip,
            relationship advice, or sports), respond with:
            "I'm designed only to help with study-related academic questions."

            Be polite and professional.
            """
        ),
        ("placeholder", "{history}"),
        ("user", "{question}")
    ]
)

llm = ChatGroq(api_key = groq_api_key, model="openai/gpt-oss-20b")
chain = prompt | llm

def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history

@app.get("/") 
def home():
    return {"message": "Welcome to the AI Study Assistant API!"}

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id)
    response = chain.invoke({"history": history, "question": request.question})

    collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.question,
        "timestamp": datetime.now(UTC)
    })

    collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.now(UTC)
    })

    return {"response" : response.content}