#!/usr/bin/env python3
"""
Legacy 20 Questions game - now using the new 20Q-ANN architecture.
This file provides backward compatibility with the original interface.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from twentyq_ann import TwentyQuestionsANN, QuestionManager
from twentyq_ann.questions import Question

def main():
    """Main entry point for the legacy interface."""
    print("🎮 20Q-ANN: Advanced 20 Questions Game")
    print("=" * 50)
    
    # Example objects and questions (same as original)
    objects = ["Dog", "Cat", "Car", "Apple", "Lion"]
    questions = [
        Question("Is it alive?", "biology"),
        Question("Is it a pet?", "pets"),
        Question("Can it fly?", "movement"),
        Question("Can you eat it?", "food"),
        Question("Is it used for transport?", "transport")
    ]
    
    # Create question manager
    question_manager = QuestionManager(questions)
    
    # Create game instance
    game = TwentyQuestionsANN(
        objects=objects,
        question_manager=question_manager,
        confidence_threshold=1.0,
        learning_rate=0.1,
        max_questions=20
    )
    
    print("Think of an object from this list:")
    print(", ".join(objects))
    print("\\nAvailable answers: yes, no, unknown, sometimes, usually, rarely, maybe, probably, probably_not, depends, irrelevant")
    print("Type 'quit' to exit.\\n")
    
    # Game loop
    while True:
        # Get next question
        question_idx = game.get_next_question()
        
        if question_idx is None:
            # Time to guess
            guess, confidence = game.get_best_guess()
            print(f"\\n🤔 My guess: {guess} (confidence: {confidence:.2f})")
            
            while True:
                result = input("Am I correct? (yes/no): ").strip().lower()
                if result in ['yes', 'y', 'correct', 'right']:
                    print("🎉 I win! Thanks for playing!")
                    game.update_weights(guess, correct=True)
                    return
                elif result in ['no', 'n', 'wrong', 'incorrect']:
                    actual = input("What was the correct answer? ").strip()
                    if actual in [obj.lower() for obj in objects]:
                        # Find the actual object
                        for obj in objects:
                            if obj.lower() == actual.lower():
                                game.update_weights(obj, correct=False)
                                print(f"Thanks! I'll learn from this. The answer was: {obj}")
                                break
                    else:
                        print("I don't recognize that object. I'll try to learn anyway.")
                        game.update_weights(guess, correct=False)
                    return
                else:
                    print("Please answer 'yes' or 'no'")
        
        # Ask question
        question_text = game.question_manager.get_question_text(question_idx)
        print(f"Question {game.question_count + 1}: {question_text}")
        
        while True:
            answer = input("Answer: ").strip().lower()
            
            if answer in ['quit', 'exit', 'q']:
                print("Thanks for playing!")
                return
            elif answer == 'help':
                print(game.question_manager.get_answer_help())
                continue
            elif game.question_manager.validate_answer(answer):
                game.submit_answer(question_idx, answer)
                break
            else:
                print("Invalid answer. Type 'help' for available options.")

if __name__ == '__main__':
    main()
