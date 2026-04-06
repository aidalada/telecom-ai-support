from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from src.analyzer import analyze_comment
from src.responder import generate_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telecom AI Support API",
    description="Интеллектуальная система анализа обращений клиентов и генерации автоответов.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommentRequest(BaseModel):
    text: str = Field(..., description="Текст комментария клиента", example="Ужасный интернет, постоянно пропадает!")

class AnalysisResult(BaseModel):
    text: str
    language: str
    moderation_verdict: str
    sentiment: str
    comment_type: str

class CommentResponse(BaseModel):
    analysis: AnalysisResult
    suggested_answer: str


@app.get("/health", tags=["System"])
def health_check():
    """Проверка статуса сервиса"""
    return {"status": "ok", "message": "API is running and models are loaded."}


@app.post("/api/v1/analyze", response_model=CommentResponse, tags=["AI Support"])
def process_comment(request: CommentRequest):
    """
    Принимает текст, прогоняет через NLP-пайплайн и возвращает анализ вместе с готовым ответом.
    """
    try:
        logger.info(f"Получен запрос на анализ: {request.text[:50]}...")
        
        analysis_data = analyze_comment(request.text)
        
        auto_reply = generate_response(analysis_data)
        
        return CommentResponse(
            analysis=AnalysisResult(**analysis_data),
            suggested_answer=auto_reply
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке комментария: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при анализе текста.")