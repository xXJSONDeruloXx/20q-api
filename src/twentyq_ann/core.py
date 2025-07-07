"""
Core ANN implementation for 20 Questions game.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import logging

from .io import WeightIO
from .questions import QuestionManager, AnswerType
from .demographics import DemographicManager, Cohort

logger = logging.getLogger(__name__)


class TwentyQuestionsANN:
    """
    Core ANN implementation for 20 Questions game.
    
    Uses a (n_objects × n_questions) weight matrix to:
    1. Rank objects based on answers (Answer→Object mode)
    2. Rank questions based on top candidates (Object→Question mode)
    """
    
    def __init__(
        self,
        objects: List[str],
        question_manager: QuestionManager,
        demographic_manager: Optional[DemographicManager] = None,
        weights_file: Optional[Union[str, Path]] = None,
        cohort: Optional[Cohort] = None,
        confidence_threshold: float = 1.0,
        learning_rate: float = 0.1,
        max_questions: int = 20,
    ):
        """
        Initialize the 20Q ANN.
        
        Args:
            objects: List of target objects
            question_manager: Manager for questions and answer mapping
            demographic_manager: Optional demographic segmentation manager
            weights_file: Path to weights file (JSON or binary)
            cohort: Demographic cohort for weight selection
            confidence_threshold: Minimum score gap for confident guess
            learning_rate: Learning rate for weight updates
            max_questions: Maximum questions per game
        """
        self.objects = objects
        self.question_manager = question_manager
        self.demographic_manager = demographic_manager
        self.n_obj = len(objects)
        self.n_q = len(question_manager.questions)
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        self.max_questions = max_questions
        self.cohort = cohort
        
        # Initialize weights
        self.weights_file = weights_file
        self.weights = self._load_weights()
        
        # Game state
        self.reset_game()
    
    def _load_weights(self) -> np.ndarray:
        """Load weights from file or initialize randomly."""
        if self.weights_file and Path(self.weights_file).exists():
            try:
                return WeightIO.load_weights(self.weights_file)
            except Exception as e:
                logger.warning(f"Failed to load weights from {self.weights_file}: {e}")
        
        # Try to load cohort-specific weights
        if self.cohort and self.demographic_manager:
            try:
                cohort_weights = self.demographic_manager.load_cohort_weights(
                    self.cohort, self.n_obj, self.n_q
                )
                if cohort_weights is not None:
                    return cohort_weights
            except Exception as e:
                logger.warning(f"Failed to load cohort weights: {e}")
        
        # Initialize random weights
        logger.info("Initializing random weights")
        return np.random.uniform(-1, 1, (self.n_obj, self.n_q))
    
    def save_weights(self, filename: Optional[Union[str, Path]] = None) -> None:
        """Save current weights to file."""
        target_file = filename or self.weights_file
        if target_file:
            WeightIO.save_weights(self.weights, target_file)
    
    def reset_game(self) -> None:
        """Reset game state for new session."""
        self.answers: Dict[int, float] = {}
        self.asked_questions: set = set()
        self.question_count = 0
        self.game_over = False
        self.final_guess: Optional[str] = None
        self.confidence_score = 0.0
    
    def rank_objects(self, answers: Optional[Dict[int, float]] = None) -> Tuple[List[int], np.ndarray]:
        """
        Rank objects based on answers (Answer→Object mode).
        
        Args:
            answers: Question index → answer weight mapping
            
        Returns:
            Tuple of (sorted object indices, scores)
        """
        if answers is None:
            answers = self.answers
            
        scores = np.zeros(self.n_obj)
        
        for question_idx, answer_weight in answers.items():
            if question_idx < self.n_q:
                scores += answer_weight * self.weights[:, question_idx]
        
        # Sort by descending score
        ranked_indices = list(np.argsort(-scores))
        return ranked_indices, scores
    
    def rank_questions(
        self, 
        top_objects: List[int], 
        strategy: str = "balanced_margin"
    ) -> Tuple[int, List[float]]:
        """
        Rank questions based on top candidates (Object→Question mode).
        
        Args:
            top_objects: Indices of top-ranked objects
            strategy: "balanced_margin" or "confirm_top"
            
        Returns:
            Tuple of (best question index, margins/scores)
        """
        if strategy == "balanced_margin":
            return self._balanced_margin_strategy(top_objects)
        elif strategy == "confirm_top":
            return self._confirm_top_strategy(top_objects)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _balanced_margin_strategy(self, top_objects: List[int]) -> Tuple[int, List[float]]:
        """Find question with most balanced yes/no split."""
        margins = []
        
        for q_idx in range(self.n_q):
            if q_idx in self.asked_questions:
                margins.append(float('inf'))  # Skip asked questions
                continue
                
            weights = self.weights[top_objects, q_idx]
            pos_sum = np.sum(weights[weights >= 0])
            neg_sum = -np.sum(weights[weights < 0])
            margin = abs(pos_sum - neg_sum)
            margins.append(margin)
        
        best_q = int(np.argmin(margins))
        return best_q, margins
    
    def _confirm_top_strategy(self, top_objects: List[int]) -> Tuple[int, List[float]]:
        """Find question that best confirms top candidate."""
        if not top_objects:
            return self._balanced_margin_strategy(top_objects)
            
        top_candidate = top_objects[0]
        scores = []
        
        for q_idx in range(self.n_q):
            if q_idx in self.asked_questions:
                scores.append(float('-inf'))  # Skip asked questions
                continue
                
            # Score based on how well this question distinguishes top candidate
            top_weight = self.weights[top_candidate, q_idx]
            other_weights = self.weights[top_objects[1:], q_idx] if len(top_objects) > 1 else np.array([])
            
            if len(other_weights) == 0:
                score = abs(top_weight)
            else:
                score = abs(top_weight - np.mean(other_weights))
            scores.append(score)
        
        best_q = int(np.argmax(scores))
        return best_q, scores
    
    def get_next_question(self, strategy: str = "balanced_margin") -> Optional[int]:
        """
        Get the next best question to ask.
        
        Args:
            strategy: Question selection strategy
            
        Returns:
            Question index or None if no more questions
        """
        if self.question_count >= self.max_questions:
            return None
            
        if len(self.asked_questions) >= self.n_q:
            return None
            
        # If no answers yet, ask first question
        if not self.answers:
            return 0
            
        # Get top candidates
        ranked_objects, scores = self.rank_objects()
        top_objects = ranked_objects[:min(5, len(ranked_objects))]
        
        # Check if confident enough to guess
        if len(ranked_objects) > 1:
            score_gap = scores[ranked_objects[0]] - scores[ranked_objects[1]]
            if score_gap > self.confidence_threshold:
                return None  # Ready to guess
        
        # Find best question
        best_q, _ = self.rank_questions(top_objects, strategy)
        
        # Ensure we don't repeat questions
        if best_q in self.asked_questions:
            for q_idx in range(self.n_q):
                if q_idx not in self.asked_questions:
                    return q_idx
            return None
            
        return best_q
    
    def submit_answer(self, question_idx: int, answer: Union[str, AnswerType]) -> None:
        """
        Submit an answer to a question.
        
        Args:
            question_idx: Index of the question
            answer: Answer string or AnswerType
        """
        if isinstance(answer, str):
            answer_type = AnswerType.from_string(answer)
        else:
            answer_type = answer
            
        answer_weight = self.question_manager.get_answer_weight(answer_type)
        
        self.answers[question_idx] = answer_weight
        self.asked_questions.add(question_idx)
        self.question_count += 1
    
    def get_best_guess(self) -> Tuple[str, float]:
        """
        Get the best guess based on current answers.
        
        Returns:
            Tuple of (object name, confidence score)
        """
        ranked_objects, scores = self.rank_objects()
        best_idx = ranked_objects[0]
        
        # Calculate confidence
        if len(ranked_objects) > 1:
            self.confidence_score = scores[best_idx] - scores[ranked_objects[1]]
        else:
            self.confidence_score = scores[best_idx]
            
        return self.objects[best_idx], self.confidence_score
    
    def update_weights(self, target_object: str, correct: bool) -> None:
        """
        Update weights based on game outcome using Hebbian learning.
        
        Args:
            target_object: The actual target object
            correct: Whether the guess was correct
        """
        try:
            target_idx = self.objects.index(target_object)
        except ValueError:
            logger.error(f"Target object '{target_object}' not found in objects list")
            return
            
        # Update weights for each answered question
        for question_idx, answer_weight in self.answers.items():
            # Skip non-informative answers
            if abs(answer_weight) < 0.1:  # threshold for "unknown", "irrelevant", etc.
                continue
                
            # Hebbian update: increase agreeable weights, decrease disagreeable ones
            if correct:
                delta = self.learning_rate * answer_weight
            else:
                delta = -self.learning_rate * answer_weight
                
            self.weights[target_idx, question_idx] += delta
        
        # Save updated weights
        self.save_weights()
        
        logger.info(f"Updated weights for {target_object}, correct={correct}")
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state for API/serialization."""
        return {
            "answers": self.answers,
            "asked_questions": list(self.asked_questions),
            "question_count": self.question_count,
            "game_over": self.game_over,
            "final_guess": self.final_guess,
            "confidence_score": self.confidence_score,
            "max_questions": self.max_questions,
        }
    
    def load_game_state(self, state: Dict[str, Any]) -> None:
        """Load game state from serialized data."""
        self.answers = state.get("answers", {})
        self.asked_questions = set(state.get("asked_questions", []))
        self.question_count = state.get("question_count", 0)
        self.game_over = state.get("game_over", False)
        self.final_guess = state.get("final_guess")
        self.confidence_score = state.get("confidence_score", 0.0)
        self.max_questions = state.get("max_questions", 20)
