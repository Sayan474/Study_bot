from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["studybot"]
collection = db["chats"]

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b"
)

# Request model
class ChatRequest(BaseModel):
    user_id: str
    question: str

@app.post("/chat")
def chat(request: ChatRequest):

    # Fetch user history
    user_data = collection.find_one({"user_id": request.user_id})
    messages = []

    if user_data:
        for msg in user_data["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

    # Add new question
    messages.append(HumanMessage(content=request.question))

    # Get response from LLM
    response = llm.invoke(messages)
    answer = response.content

    # Store updated conversation
    updated_messages = user_data["messages"] if user_data else []
    updated_messages.append({"role": "user", "content": request.question})
    updated_messages.append({"role": "assistant", "content": answer})

    collection.update_one(
        {"user_id": request.user_id},
        {"$set": {"messages": updated_messages}},
        upsert=True
    )

    return {"answer": answer}