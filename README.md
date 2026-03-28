📘 AI Study Assistant Chatbot

An AI-powered Study Assistant chatbot built using FastAPI, LangChain, and MongoDB that answers academic questions and maintains conversation memory for each user.

🚀 Features

📚 Answers study-related academic questions only

🧠 Maintains persistent chat memory using MongoDB

👥 Supports multiple users via user_id

🔄 Provides context-aware responses

🌐 Deployed as a REST API

⚡ Powered by Groq LLM (LLaMA models)

🏗️ Tech Stack

Backend: FastAPI
LLM Integration: LangChain
Database: MongoDB Atlas
Deployment: Render
Language: Python

⚙️ Setup Instructions (Local)
1️⃣ Clone the Repository
git clone https://github.com/yourusername/study-bot-memory.git
cd study-bot-memory
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Configure Environment Variables

Create a .env file:

MONGO_URI=your_mongodb_connection_string
GROQ_API_KEY=your_groq_api_key
5️⃣ Run the Server
uvicorn app:app --reload

Open:

http://127.0.0.1:8000/docs
🔗 API Usage
Endpoint
POST /chat
Request
{
  "user_id": "user1",
  "question": "What is recursion?"
}
Response
{
  "answer": "Recursion is a programming technique..."
}
🧠 Memory Implementation
Chat history is stored in MongoDB using user_id
Each user has a separate conversation history
Previous messages are retrieved and sent to the LLM
Enables context-aware responses

🎯 Domain Restriction

The chatbot is designed to answer only academic questions.

Allowed:

Mathematics
Programming
Science subjects
Engineering topics

If a non-academic question is asked, it responds:

"I'm designed only to help with study-related academic questions."

🌐 Deployment

The project is deployed on Render.

🔗 Hosted API:
[https://study-bot-ffw1.onrender.com](https://study-bot-ffw1.onrender.com/)
📄 Swagger Docs:
[https://your-render-link.onrender.com/docs](https://study-bot-ffw1.onrender.com/docs)
