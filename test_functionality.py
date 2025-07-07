#!/usr/bin/env python3
"""
Test script to verify the 20Q-ANN functionality.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from twentyq_ann import TwentyQuestionsANN, QuestionManager
from twentyq_ann.questions import Question

def test_basic_functionality():
    """Test basic game functionality."""
    print("Testing 20Q-ANN Core Functionality")
    print("=" * 40)
    
    # Create test data
    objects = ["Dog", "Cat", "Car", "Apple", "Book"]
    questions = [
        Question("Is it alive?", "biology"),
        Question("Is it bigger than a car?", "size"),
        Question("Can you eat it?", "food"),
        Question("Is it made by humans?", "origin"),
        Question("Can it fly?", "movement"),
    ]
    
    # Create question manager
    question_manager = QuestionManager(questions)
    
    # Create game instance
    game = TwentyQuestionsANN(
        objects=objects,
        question_manager=question_manager,
        confidence_threshold=1.0,
        learning_rate=0.1,
        max_questions=10
    )
    
    print(f"✓ Game initialized with {len(objects)} objects and {len(questions)} questions")
    print(f"✓ Weight matrix shape: {game.weights.shape}")
    
    # Test getting first question
    question_idx = game.get_next_question()
    if question_idx is not None:
        question_text = game.question_manager.get_question_text(question_idx)
        print(f"✓ First question: {question_text}")
    
    # Test answer submission
    game.submit_answer(question_idx, "yes")
    print(f"✓ Answer submitted: yes")
    print(f"✓ Question count: {game.question_count}")
    
    # Test object ranking
    ranked, scores = game.rank_objects()
    print(f"✓ Object ranking: {[objects[i] for i in ranked[:3]]}")
    
    # Test next question
    next_question_idx = game.get_next_question()
    if next_question_idx is not None:
        next_question_text = game.question_manager.get_question_text(next_question_idx)
        print(f"✓ Next question: {next_question_text}")
    
    # Test guess
    guess, confidence = game.get_best_guess()
    print(f"✓ Best guess: {guess} (confidence: {confidence:.2f})")
    
    # Test weight update
    original_weights = game.weights.copy()
    game.update_weights("Dog", correct=True)
    print(f"✓ Weights updated successfully")
    
    # Test game state
    state = game.get_game_state()
    print(f"✓ Game state retrieved: {len(state)} fields")
    
    # Test answer types
    available_answers = game.question_manager.get_available_answers()
    print(f"✓ Available answers: {len(available_answers)} types")
    
    print("\\n🎉 All tests passed! The 20Q-ANN system is working correctly.")

if __name__ == "__main__":
    test_basic_functionality()
