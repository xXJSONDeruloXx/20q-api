"""
FastAPI REST interface for the 20Q-ANN game.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from pathlib import Path

from .core import TwentyQuestionsANN
from .questions import QuestionManager, AnswerType
from .demographics import DemographicManager

logger = logging.getLogger(__name__)

# Pydantic models
class GameConfig(BaseModel):
    """Configuration for starting a new game."""
    objects: Optional[List[str]] = None
    questions_file: Optional[str] = None
    weights_file: Optional[str] = None
    cohort_id: Optional[str] = None
    max_questions: int = Field(default=20, ge=1, le=50)
    confidence_threshold: float = Field(default=1.0, ge=0.1, le=10.0)
    learning_rate: float = Field(default=0.1, ge=0.01, le=1.0)

class GameState(BaseModel):
    """Current state of a game session."""
    session_id: str
    question_count: int
    max_questions: int
    game_over: bool
    current_question: Optional[str] = None
    question_index: Optional[int] = None
    available_answers: List[str]
    confidence_score: float = 0.0
    final_guess: Optional[str] = None

class AnswerRequest(BaseModel):
    """Request to submit an answer."""
    session_id: str
    answer: str

class GuessResult(BaseModel):
    """Result of a guess."""
    session_id: str
    guess: str
    confidence: float
    correct: bool

class GameStats(BaseModel):
    """Game statistics."""
    total_games: int
    total_questions_asked: int
    average_questions_per_game: float
    success_rate: float
    cohort_stats: Dict[str, Any]

# Global game sessions storage
# In production, this would be a database or Redis
game_sessions: Dict[str, TwentyQuestionsANN] = {}

# FastAPI app
app = FastAPI(
    title="20Q-ANN API",
    description="REST API for the 20 Questions ANN game",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencies
def get_demographic_manager() -> DemographicManager:
    """Get demographic manager instance."""
    return DemographicManager()

def get_question_manager(questions_file: Optional[str] = None) -> QuestionManager:
    """Get question manager instance."""
    if questions_file and Path(questions_file).exists():
        return QuestionManager.from_json(Path(questions_file))
    
    # Default questions
    from .questions import Question
    default_questions = [
        Question("Is it alive?", category="biology"),
        Question("Is it bigger than a car?", category="size"),
        Question("Can you eat it?", category="food"),
        Question("Is it made by humans?", category="origin"),
        Question("Can it fly?", category="movement"),
        Question("Is it found in a house?", category="location"),
        Question("Is it electronic?", category="technology"),
        Question("Is it used for entertainment?", category="purpose"),
        Question("Is it soft?", category="texture"),
        Question("Does it have legs?", category="anatomy"),
        Question("Is it expensive?", category="value"),
        Question("Is it colorful?", category="appearance"),
        Question("Can you hold it in your hand?", category="size"),
        Question("Is it used for transportation?", category="transport"),
        Question("Does it make noise?", category="sound"),
    ]
    return QuestionManager(default_questions)

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "20Q-ANN API", "version": "0.1.0"}

@app.post("/game/start", response_model=GameState)
async def start_game(
    config: GameConfig,
    demographic_manager: DemographicManager = Depends(get_demographic_manager)
):
    """Start a new game session."""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Default objects if not provided
        objects = config.objects or [
            "Dog", "Cat", "Car", "Apple", "Lion", "Airplane", "Computer",
            "Guitar", "Pizza", "Ocean", "Mountain", "Book", "Phone", "Tree",
            "House", "Bicycle", "Coffee", "Moon", "Fire", "Ice"
        ]
        
        # Get question manager
        question_manager = get_question_manager(config.questions_file)
        
        # Get cohort
        cohort = None
        if config.cohort_id:
            cohort = demographic_manager.get_cohort(config.cohort_id)
        
        # Create game instance
        game = TwentyQuestionsANN(
            objects=objects,
            question_manager=question_manager,
            demographic_manager=demographic_manager,
            weights_file=config.weights_file,
            cohort=cohort,
            confidence_threshold=config.confidence_threshold,
            learning_rate=config.learning_rate,
            max_questions=config.max_questions
        )
        
        # Store session
        game_sessions[session_id] = game
        
        # Get first question
        first_question_idx = game.get_next_question()
        first_question = None
        if first_question_idx is not None:
            first_question = game.question_manager.get_question_text(first_question_idx)
        
        return GameState(
            session_id=session_id,
            question_count=0,
            max_questions=config.max_questions,
            game_over=False,
            current_question=first_question,
            question_index=first_question_idx,
            available_answers=game.question_manager.get_available_answers(),
            confidence_score=0.0
        )
        
    except Exception as e:
        logger.error(f"Error starting game: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/game/answer", response_model=GameState)
async def submit_answer(request: AnswerRequest):
    """Submit an answer to the current question."""
    try:
        # Get game session
        if request.session_id not in game_sessions:
            raise HTTPException(status_code=404, detail="Game session not found")
        
        game = game_sessions[request.session_id]
        
        # Validate answer
        if not game.question_manager.validate_answer(request.answer):
            raise HTTPException(status_code=400, detail="Invalid answer")
        
        # Get current question
        current_question_idx = game.get_next_question()
        if current_question_idx is None:
            raise HTTPException(status_code=400, detail="No question to answer")
        
        # Submit answer
        game.submit_answer(current_question_idx, request.answer)
        
        # Get next question or prepare guess
        next_question_idx = game.get_next_question()
        next_question = None
        final_guess = None
        
        if next_question_idx is not None:
            next_question = game.question_manager.get_question_text(next_question_idx)
        else:
            # Ready to guess
            final_guess, confidence = game.get_best_guess()
            game.game_over = True
            game.final_guess = final_guess
            game.confidence_score = confidence
        
        return GameState(
            session_id=request.session_id,
            question_count=game.question_count,
            max_questions=game.max_questions,
            game_over=game.game_over,
            current_question=next_question,
            question_index=next_question_idx,
            available_answers=game.question_manager.get_available_answers(),
            confidence_score=game.confidence_score,
            final_guess=final_guess
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/game/guess", response_model=GuessResult)
async def submit_guess_result(request: GuessResult):
    """Submit the result of a guess for learning."""
    try:
        # Get game session
        if request.session_id not in game_sessions:
            raise HTTPException(status_code=404, detail="Game session not found")
        
        game = game_sessions[request.session_id]
        
        # Update weights based on result
        if request.correct:
            game.update_weights(request.guess, correct=True)
        else:
            game.update_weights(request.guess, correct=False)
        
        # Clean up session
        del game_sessions[request.session_id]
        
        return request
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing guess result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/game/state/{session_id}", response_model=GameState)
async def get_game_state(session_id: str):
    """Get the current state of a game session."""
    try:
        if session_id not in game_sessions:
            raise HTTPException(status_code=404, detail="Game session not found")
        
        game = game_sessions[session_id]
        
        # Get current question
        current_question_idx = game.get_next_question()
        current_question = None
        if current_question_idx is not None:
            current_question = game.question_manager.get_question_text(current_question_idx)
        
        return GameState(
            session_id=session_id,
            question_count=game.question_count,
            max_questions=game.max_questions,
            game_over=game.game_over,
            current_question=current_question,
            question_index=current_question_idx,
            available_answers=game.question_manager.get_available_answers(),
            confidence_score=game.confidence_score,
            final_guess=game.final_guess
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/game/{session_id}")
async def end_game(session_id: str):
    """End a game session."""
    try:
        if session_id not in game_sessions:
            raise HTTPException(status_code=404, detail="Game session not found")
        
        del game_sessions[session_id]
        return {"message": "Game session ended"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending game: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cohorts")
async def list_cohorts(
    demographic_manager: DemographicManager = Depends(get_demographic_manager)
):
    """List available demographic cohorts."""
    try:
        cohorts = demographic_manager.list_cohorts()
        return [
            {
                "id": cohort.id,
                "name": cohort.name,
                "type": cohort.type.value,
                "description": cohort.description
            }
            for cohort in cohorts
        ]
        
    except Exception as e:
        logger.error(f"Error listing cohorts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cohorts/{cohort_id}")
async def get_cohort_stats(
    cohort_id: str,
    demographic_manager: DemographicManager = Depends(get_demographic_manager)
):
    """Get statistics for a specific cohort."""
    try:
        stats = demographic_manager.get_cohort_stats(cohort_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Cohort not found")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cohort stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=GameStats)
async def get_game_stats():
    """Get overall game statistics."""
    try:
        # This would typically come from a database
        # For now, return mock stats
        return GameStats(
            total_games=0,
            total_questions_asked=0,
            average_questions_per_game=0.0,
            success_rate=0.0,
            cohort_stats={}
        )
        
    except Exception as e:
        logger.error(f"Error getting game stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(game_sessions)}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
