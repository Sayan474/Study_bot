import datetime
import os
import hashlib
import secrets
import re
import html
from uuid import uuid4
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime, UTC
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

client = MongoClient(mongo_uri)
db = client["chat"]
chat_collection = db["users"]
users_collection = db["app_users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
            The currently authenticated user's registered username is: {registered_username}

            You are allowed to:
            - Greet users politely
            - Explain who you are
            - Explain what you can do
            - Ask for clarification if needed
            - Answer academic questions
            - Handle personalization requests about the user's name

            Name handling rule:
            - If the user asks for their name (for example: "what is my name?"), reply with the most recent name the user provided in chat.
            - If the user has not provided a name in chat, use the registered username.

            Allowed academic topics include:
            Mathematics, Computer Science, Programming,
            Physics, Chemistry, Biology, Engineering,
            History, Geography, Economics, Exam preparation.

                        Formatting and readability rules:
                        - Output plain text only.
                        - Do not use Markdown syntax (no **, #, tables with |, or fenced code blocks).
                        - Do not use HTML entities like &nbsp;.
                        - Use clear line breaks between sections.
                        - For programming concepts or algorithms, use this exact readable layout:
                            Overview:
                            Steps:
                            Code (language name):
                            Complexity:
                            Example:
                        - Keep explanations concise and human-friendly.
                        - Never output one long mixed paragraph for coding answers.

                        Math formatting rule:
                        - Write mathematical expressions in readable text form (e.g., sqrt(x), x^2, sin(x)).
                        - Do not use LaTeX delimiters like \[ \] or \( \).

            If a user asks about topics unrelated to academics
            (such as entertainment, jokes, politics, gossip,
            relationship advice, or sports), respond with:
            "I'm designed only to help with study-related academic questions."

            Exception:
            Name and basic personalization requests are allowed even if they are not academic.

            Be polite and professional.
            """
        ),
        ("placeholder", "{history}"),
        ("user", "{question}")
    ]
)

llm = ChatGroq(api_key = groq_api_key, model="openai/gpt-oss-20b")
chain = prompt | llm


def hash_password(password: str, salt: str) -> str:
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    )
    return password_hash.hex()


def build_password_record(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    return salt, password_hash


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    return hash_password(password, salt) == expected_hash


def sanitize_assistant_response(text: str) -> str:
    if not text:
        return ""

    # Normalize escaped entities and line endings first.
    cleaned = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    # Remove fenced code wrappers if present.
    cleaned = re.sub(r"```[a-zA-Z0-9_+-]*", "", cleaned)
    cleaned = cleaned.replace("```", "")

    output_lines = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            output_lines.append("")
            continue

        # Drop markdown horizontal rules.
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Convert markdown table rows to readable plain text.
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            # Skip separator rows such as |---|---|
            if all(re.fullmatch(r"[:\- ]*", cell) for cell in cells):
                continue
            output_lines.append(" - ".join(cell for cell in cells if cell))
            continue

        # Remove heading markers and markdown emphasis tokens.
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = line.replace("**", "").replace("__", "")
        line = line.replace("`", "")

        output_lines.append(line)

    # Collapse repeated blank lines for cleaner rendering.
    normalized = "\n".join(output_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()

def get_history(user_id):
    chats = chat_collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history

@app.get("/") 
def home():
    return {"message": "Welcome to the AI Study Assistant API!"}


@app.post("/auth/signup")
def signup(request: SignupRequest):
    existing_user = users_collection.find_one({"email": request.email.lower()})
    if existing_user:
        raise HTTPException(status_code=409, detail="Email is already registered")

    salt, password_hash = build_password_record(request.password)
    user_id = str(uuid4())

    users_collection.insert_one(
        {
            "user_id": user_id,
            "username": request.username.strip(),
            "email": request.email.lower().strip(),
            "password_salt": salt,
            "password_hash": password_hash,
            "created_at": datetime.now(UTC),
        }
    )

    return {
        "message": "Signup successful",
        "user": {
            "user_id": user_id,
            "username": request.username.strip(),
            "email": request.email.lower().strip(),
        },
    }


@app.post("/auth/login")
def login(request: LoginRequest):
    user = users_collection.find_one({"email": request.email.lower().strip()})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(request.password, user["password_salt"], user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "message": "Login successful",
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
        },
    }


@app.get("/chat/history/{user_id}")
def chat_history(user_id: str):
    user_exists = users_collection.find_one({"user_id": user_id})
    if not user_exists:
        raise HTTPException(status_code=404, detail="User not found")

    chats = list(chat_collection.find({"user_id": user_id}).sort("timestamp", 1))
    return {
        "user_id": user_id,
        "history": [
            {
                "role": chat["role"],
                "message": chat["message"],
                "timestamp": chat.get("timestamp"),
            }
            for chat in chats
        ],
    }


@app.delete("/chat/history/{user_id}")
def clear_chat_history(user_id: str):
    user_exists = users_collection.find_one({"user_id": user_id})
    if not user_exists:
        raise HTTPException(status_code=404, detail="User not found")

    result = chat_collection.delete_many({"user_id": user_id})
    return {
        "message": "Chat history cleared",
        "deleted_count": result.deleted_count,
    }

@app.post("/chat")
def chat(request: ChatRequest):
    user_exists = users_collection.find_one({"user_id": request.user_id})
    if not user_exists:
        raise HTTPException(status_code=404, detail="User not found")

    history = get_history(request.user_id)
    response = chain.invoke(
        {
            "registered_username": user_exists.get("username", "student"),
            "history": history,
            "question": request.question,
        }
    )
    assistant_message = sanitize_assistant_response(response.content)

    chat_collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.question,
        "timestamp": datetime.now(UTC)
    })

    chat_collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": assistant_message,
        "timestamp": datetime.now(UTC)
    })

    return {"response" : assistant_message}