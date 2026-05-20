import os
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

# ----------------------------------------
# Load ENV
# ----------------------------------------
load_dotenv()

# ----------------------------------------
# OpenAI Client
# ----------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_FREE")
)

# ----------------------------------------
# FastAPI App
# ----------------------------------------
app = FastAPI(
    title="Hybrid RAG API",
    version="1.0"
)

# ----------------------------------------
# CORS
# ----------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Load Embeddings
# ----------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ----------------------------------------
# Load FAISS Vector DB
# ----------------------------------------
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# ----------------------------------------
# Request Model
# ----------------------------------------
class QuestionRequest(BaseModel):

    question: str
    language: str = "English"

# ----------------------------------------
# Health Check
# ----------------------------------------
@app.get("/")
def home():

    return {
        "status": "success",
        "message": "Hybrid RAG API Running"
    }

# ----------------------------------------
# Ask Endpoint
# ----------------------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        query = request.question
        language = request.language

        # ----------------------------------------
        # Language Handling
        # ----------------------------------------
        if language.lower() == "telugu":

            response_language = "Telugu"

        else:

            response_language = "English"

        # ----------------------------------------
        # Similarity Search
        # ----------------------------------------
        results = db.similarity_search_with_score(
            query,
            k=3
        )

        context = ""
        use_rag = False

        # Lower score = better match
        SIMILARITY_THRESHOLD = 1.0

        if results:

            best_score = results[0][1]

            if best_score < SIMILARITY_THRESHOLD:

                use_rag = True

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc, score in results
                    ]
                )

        # ----------------------------------------
        # Prompt Creation
        # ----------------------------------------
        if use_rag:

            prompt = f"""
You are a helpful AI assistant.

The below CONTEXT comes from Bhagavad Gita PDF.

RULES:
1. First use the PDF context to answer.
2. If context partially contains answer, combine PDF knowledge with your own knowledge.
3. If context is insufficient, use your own knowledge.
4. Never say "I don't know".
5. Keep answer within 5 to 6 lines.
6. Respond ONLY in {response_language}
7. If Telugu selected, respond in Telugu script.
8. Do not mention whether answer came from PDF or AI knowledge.

Question:
{query}

PDF Context:
{context}
"""

        else:

            prompt = f"""
You are a knowledgeable AI assistant.

The PDF does not contain relevant information for this question.

Answer using your own knowledge.

RULES:
1. Never say "I don't know".
2. Keep answer within 5 to 6 lines.
3. Respond ONLY in {response_language}
4. If Telugu selected, respond in Telugu script.
5. Keep answer simple and clear.

Question:
{query}
"""

        # ----------------------------------------
        # OpenAI API
        # ----------------------------------------
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart hybrid RAG assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        # ----------------------------------------
        # Final Response
        # ----------------------------------------
        return {

            "success": True,
            "question": query,
            "answer": answer,
            "source": (
                "PDF + AI Knowledge"
                if use_rag
                else "AI General Knowledge"
            ),
            "language": response_language
        }

    except Exception as e:

        return {

            "success": False,
            "error": str(e)
        }