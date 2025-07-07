"""
Tests for the core TwentyQuestionsANN class.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from twentyq_ann.core import TwentyQuestionsANN
from twentyq_ann.questions import QuestionManager, Question, AnswerType
from twentyq_ann.demographics import DemographicManager


@pytest.fixture
def sample_objects():
    """Sample objects for testing."""
    return ["Dog", "Cat", "Car", "Apple", "Book"]


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        Question("Is it alive?", "biology"),
        Question("Is it bigger than a car?", "size"),
        Question("Can you eat it?", "food"),
        Question("Is it made by humans?", "origin"),
        Question("Can it fly?", "movement"),
    ]


@pytest.fixture
def question_manager(sample_questions):
    """Question manager fixture."""
    return QuestionManager(sample_questions)


@pytest.fixture
def demographic_manager():
    """Demographic manager fixture."""
    return DemographicManager()


@pytest.fixture
def game_instance(sample_objects, question_manager, demographic_manager):
    """Game instance fixture."""
    return TwentyQuestionsANN(
        objects=sample_objects,
        question_manager=question_manager,
        demographic_manager=demographic_manager,
        confidence_threshold=1.0,
        learning_rate=0.1,
        max_questions=10
    )


class TestTwentyQuestionsANN:
    """Test cases for TwentyQuestionsANN class."""
    
    def test_initialization(self, game_instance):
        """Test proper initialization."""
        assert len(game_instance.objects) == 5
        assert game_instance.n_obj == 5
        assert game_instance.n_q == 5
        assert game_instance.weights.shape == (5, 5)
        assert game_instance.confidence_threshold == 1.0
        assert game_instance.learning_rate == 0.1
        assert game_instance.max_questions == 10
    
    def test_reset_game(self, game_instance):
        """Test game reset functionality."""
        # Add some state
        game_instance.answers = {0: 1.0, 1: -1.0}
        game_instance.asked_questions = {0, 1}
        game_instance.question_count = 2
        game_instance.game_over = True
        
        # Reset
        game_instance.reset_game()
        
        assert game_instance.answers == {}
        assert game_instance.asked_questions == set()
        assert game_instance.question_count == 0
        assert game_instance.game_over == False
        assert game_instance.final_guess is None
        assert game_instance.confidence_score == 0.0
    
    def test_rank_objects(self, game_instance):
        """Test object ranking."""
        # Set up test weights
        game_instance.weights = np.array([
            [1.0, -1.0, 0.0, 1.0, 0.0],  # Dog
            [1.0, -1.0, 0.0, 0.0, 0.0],  # Cat
            [-1.0, 1.0, 0.0, 1.0, 0.0],  # Car
            [1.0, -1.0, 1.0, 0.0, 0.0],  # Apple
            [-1.0, -1.0, 0.0, 1.0, 0.0], # Book
        ])
        
        # Test with answers
        answers = {0: 1.0, 1: -1.0}  # Yes to "Is it alive?", No to "Is it bigger than a car?"
        
        ranked, scores = game_instance.rank_objects(answers)
        
        assert len(ranked) == 5
        assert len(scores) == 5
        
        # Dog and Cat should score higher (alive and not bigger than car)
        assert scores[0] > scores[2]  # Dog > Car
        assert scores[1] > scores[2]  # Cat > Car
    
    def test_rank_questions_balanced_margin(self, game_instance):
        """Test question ranking with balanced margin strategy."""
        # Set up test weights
        game_instance.weights = np.array([
            [1.0, -1.0, 0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, -1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0, 0.0],
        ])
        
        top_objects = [0, 1, 3]  # Dog, Cat, Apple
        
        best_q, margins = game_instance.rank_questions(top_objects, "balanced_margin")
        
        assert 0 <= best_q < 5
        assert len(margins) == 5
        assert all(isinstance(m, (int, float)) for m in margins)
    
    def test_get_next_question(self, game_instance):
        """Test getting next question."""
        # First question should be 0
        next_q = game_instance.get_next_question()
        assert next_q == 0
        
        # After answering, should get a different question
        game_instance.submit_answer(0, "yes")
        next_q = game_instance.get_next_question()
        assert next_q != 0
        assert next_q is not None
    
    def test_submit_answer(self, game_instance):
        """Test submitting answers."""
        # Submit answer
        game_instance.submit_answer(0, "yes")
        
        assert 0 in game_instance.answers
        assert game_instance.answers[0] == 1.0
        assert 0 in game_instance.asked_questions
        assert game_instance.question_count == 1
        
        # Test with AnswerType
        game_instance.submit_answer(1, AnswerType.NO)
        assert game_instance.answers[1] == -1.0
    
    def test_get_best_guess(self, game_instance):
        """Test getting best guess."""
        # Set up some answers
        game_instance.answers = {0: 1.0, 1: -1.0}
        
        guess, confidence = game_instance.get_best_guess()
        
        assert guess in game_instance.objects
        assert isinstance(confidence, (int, float))
        assert game_instance.confidence_score == confidence
    
    def test_update_weights(self, game_instance):
        """Test weight updates."""
        # Set up initial state
        game_instance.answers = {0: 1.0, 1: -1.0}
        initial_weights = game_instance.weights.copy()
        
        # Update weights for correct guess
        game_instance.update_weights("Dog", correct=True)
        
        # Weights should have changed
        assert not np.array_equal(initial_weights, game_instance.weights)
        
        # Dog (index 0) weights should have increased for positive answers
        assert game_instance.weights[0, 0] > initial_weights[0, 0]  # yes to "Is it alive?"
        assert game_instance.weights[0, 1] < initial_weights[0, 1]  # no to "Is it bigger than a car?"
    
    def test_game_state_serialization(self, game_instance):
        """Test game state serialization."""
        # Set up some state
        game_instance.answers = {0: 1.0, 1: -1.0}
        game_instance.asked_questions = {0, 1}
        game_instance.question_count = 2
        game_instance.final_guess = "Dog"
        game_instance.confidence_score = 1.5
        
        # Get state
        state = game_instance.get_game_state()
        
        assert state["answers"] == {0: 1.0, 1: -1.0}
        assert set(state["asked_questions"]) == {0, 1}
        assert state["question_count"] == 2
        assert state["final_guess"] == "Dog"
        assert state["confidence_score"] == 1.5
        
        # Test loading state
        game_instance.reset_game()
        game_instance.load_game_state(state)
        
        assert game_instance.answers == {0: 1.0, 1: -1.0}
        assert game_instance.asked_questions == {0, 1}
        assert game_instance.question_count == 2
        assert game_instance.final_guess == "Dog"
        assert game_instance.confidence_score == 1.5
    
    def test_weights_save_load(self, game_instance):
        """Test saving and loading weights."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Modify weights
            game_instance.weights[0, 0] = 42.0
            
            # Save weights
            game_instance.save_weights(temp_path)
            
            # Create new instance and load
            new_game = TwentyQuestionsANN(
                objects=game_instance.objects,
                question_manager=game_instance.question_manager,
                weights_file=temp_path
            )
            
            # Check weights were loaded correctly
            assert new_game.weights[0, 0] == 42.0
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_confidence_threshold_guessing(self, game_instance):
        """Test confidence threshold affects guessing."""
        # Set weights to create clear winner
        game_instance.weights = np.array([
            [10.0, 0.0, 0.0, 0.0, 0.0],  # Dog - high weight
            [1.0, 0.0, 0.0, 0.0, 0.0],   # Cat - low weight
            [1.0, 0.0, 0.0, 0.0, 0.0],   # Car - low weight
            [1.0, 0.0, 0.0, 0.0, 0.0],   # Apple - low weight
            [1.0, 0.0, 0.0, 0.0, 0.0],   # Book - low weight
        ])
        
        # Answer first question positively
        game_instance.submit_answer(0, "yes")
        
        # With low threshold, should be ready to guess
        game_instance.confidence_threshold = 1.0
        next_q = game_instance.get_next_question()
        assert next_q is None  # Ready to guess
        
        # Reset and try with high threshold
        game_instance.reset_game()
        game_instance.confidence_threshold = 20.0
        game_instance.submit_answer(0, "yes")
        next_q = game_instance.get_next_question()
        assert next_q is not None  # Not ready to guess yet
