#!/usr/bin/env python3
"""
Demo script showing advanced features of the 20Q-ANN system.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from twentyq_ann import TwentyQuestionsANN, QuestionManager
from twentyq_ann.questions import Question, AnswerType
from twentyq_ann.demographics import DemographicManager, Cohort, CohortType
from twentyq_ann.io import WeightIO
import numpy as np

def demo_advanced_features():
    """Demonstrate advanced features of the 20Q-ANN system."""
    print("🚀 20Q-ANN Advanced Features Demo")
    print("=" * 50)
    
    # 1. Multi-valued Answers Demo
    print("\\n1. Multi-valued Answers")
    print("-" * 25)
    
    question_manager = QuestionManager()
    
    # Show all available answer types
    print("Available answer types:")
    for answer_type in AnswerType:
        weight = question_manager.get_answer_weight(answer_type)
        print(f"  {answer_type.value:15} → {weight:5.1f}")
    
    # 2. Question Categories Demo
    print("\\n2. Question Categories & Metadata")
    print("-" * 35)
    
    questions = [
        Question("Is it alive?", "biology", difficulty=1, tags=["living", "organic"]),
        Question("Is it bigger than a car?", "size", difficulty=1, tags=["size", "comparison"]),
        Question("Can you eat it?", "food", difficulty=1, tags=["edible", "consumption"]),
        Question("Is it electronic?", "technology", difficulty=2, tags=["electronic", "powered"]),
        Question("Is it expensive?", "value", difficulty=3, tags=["cost", "price"]),
    ]
    
    question_manager = QuestionManager(questions)
    
    # Show question stats
    stats = question_manager.get_stats()
    print(f"Total questions: {stats['total_questions']}")
    print(f"Categories: {stats['categories']}")
    print(f"Difficulties: {stats['difficulties']}")
    
    # 3. Demographic Cohorts Demo
    print("\\n3. Demographic Cohorts")
    print("-" * 25)
    
    demographic_manager = DemographicManager()
    demographic_manager.create_default_cohorts()
    
    cohorts = demographic_manager.list_cohorts()
    print(f"Available cohorts: {len(cohorts)}")
    for cohort in cohorts:
        print(f"  {cohort.name} ({cohort.type.value})")
    
    # 4. Weight Storage Demo
    print("\\n4. Weight Storage & Compression")
    print("-" * 35)
    
    # Create sample weights
    weights = np.random.uniform(-1, 1, (20, 15))  # 20 objects, 15 questions
    
    # Save in both formats
    WeightIO.save_weights(weights, "demo_weights.json")
    WeightIO.save_weights(weights, "demo_weights.bin")
    
    # Show file sizes
    json_size = Path("demo_weights.json").stat().st_size
    bin_size = Path("demo_weights.bin").stat().st_size
    compression_ratio = json_size / bin_size
    
    print(f"JSON size: {json_size:,} bytes")
    print(f"Binary size: {bin_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    
    # Clean up
    Path("demo_weights.json").unlink()
    Path("demo_weights.bin").unlink()
    
    # 5. Game Strategy Demo
    print("\\n5. Question Selection Strategies")
    print("-" * 35)
    
    objects = ["Dog", "Cat", "Car", "Apple", "Book", "Phone", "Tree", "Guitar"]
    game = TwentyQuestionsANN(
        objects=objects,
        question_manager=question_manager,
        confidence_threshold=1.0
    )
    
    # Test both strategies
    top_objects = [0, 1, 2]  # Top 3 objects
    
    balanced_q, balanced_margins = game.rank_questions(top_objects, "balanced_margin")
    confirm_q, confirm_scores = game.rank_questions(top_objects, "confirm_top")
    
    print(f"Balanced margin strategy: Question {balanced_q}")
    print(f"Confirm top strategy: Question {confirm_q}")
    
    # 6. Adaptive Learning Demo
    print("\\n6. Adaptive Learning")
    print("-" * 20)
    
    # Simulate a few learning cycles
    initial_weights = game.weights.copy()
    
    # Simulate correct answer
    game.answers = {0: 1.0, 1: -1.0}  # Yes to alive, No to bigger than car
    game.update_weights("Dog", correct=True)
    
    # Show weight changes
    weight_changes = np.abs(game.weights - initial_weights)
    total_change = np.sum(weight_changes)
    
    print(f"Total weight change: {total_change:.3f}")
    print(f"Learning rate: {game.learning_rate}")
    
    # 7. Advanced Answer Processing
    print("\\n7. Advanced Answer Processing")
    print("-" * 32)
    
    # Show answer mapping variations
    test_answers = ["yes", "y", "sometimes", "usually", "probably_not", "depends", "irrelevant"]
    
    for answer in test_answers:
        answer_type = AnswerType.from_string(answer)
        weight = question_manager.get_answer_weight(answer_type)
        print(f"  '{answer}' → {answer_type.value} → {weight:5.1f}")
    
    print("\\n🎉 Demo complete! The 20Q-ANN system offers rich functionality")
    print("   for building sophisticated question-answering games.")

if __name__ == "__main__":
    demo_advanced_features()
