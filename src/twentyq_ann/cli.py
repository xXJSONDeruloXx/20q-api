"""
CLI interface for the 20Q-ANN game.
"""

import click
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from .core import TwentyQuestionsANN
from .questions import QuestionManager, AnswerType
from .demographics import DemographicManager
from .io import WeightIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool) -> None:
    """20Q-ANN: Production-quality 20 Questions engine."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--objects', '-o', type=click.Path(exists=True), 
              help='Path to objects JSON file')
@click.option('--questions', '-q', type=click.Path(exists=True),
              help='Path to questions JSON file')
@click.option('--weights', '-w', type=click.Path(),
              help='Path to weights file')
@click.option('--cohort', '-c', type=str,
              help='Demographic cohort ID')
@click.option('--max-questions', '-m', type=int, default=20,
              help='Maximum number of questions')
@click.option('--confidence', '-t', type=float, default=1.0,
              help='Confidence threshold for guessing')
@click.option('--learning-rate', '-l', type=float, default=0.1,
              help='Learning rate for weight updates')
def play(
    objects: Optional[str],
    questions: Optional[str], 
    weights: Optional[str],
    cohort: Optional[str],
    max_questions: int,
    confidence: float,
    learning_rate: float
) -> None:
    """Play an interactive 20 Questions game."""
    
    # Load objects
    if objects:
        with open(objects, 'r') as f:
            objects_data = json.load(f)
        object_list = objects_data.get('objects', objects_data)
    else:
        # Default objects
        object_list = [
            "Dog", "Cat", "Car", "Apple", "Lion", "Airplane", "Computer",
            "Guitar", "Pizza", "Ocean", "Mountain", "Book", "Phone", "Tree",
            "House", "Bicycle", "Coffee", "Moon", "Fire", "Ice"
        ]
    
    # Load questions
    if questions:
        question_manager = QuestionManager.from_json(Path(questions))
    else:
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
        question_manager = QuestionManager(default_questions)
    
    # Load demographic manager
    demographic_manager = DemographicManager()
    
    # Get cohort
    cohort_obj = None
    if cohort:
        cohort_obj = demographic_manager.get_cohort(cohort)
        if not cohort_obj:
            click.echo(f"Warning: Cohort '{cohort}' not found")
    
    # Initialize game
    game = TwentyQuestionsANN(
        objects=object_list,
        question_manager=question_manager,
        demographic_manager=demographic_manager,
        weights_file=weights,
        cohort=cohort_obj,
        confidence_threshold=confidence,
        learning_rate=learning_rate,
        max_questions=max_questions
    )
    
    # Game loop
    click.echo("🎮 Welcome to 20Q-ANN!")
    click.echo("Think of an object and I'll try to guess it!")
    click.echo("Available answers: yes, no, unknown, sometimes, usually, rarely, maybe, probably, probably_not, depends, irrelevant")
    click.echo("Type 'help' for answer explanations or 'quit' to exit.\\n")
    
    while True:
        # Get next question
        next_q = game.get_next_question()
        
        if next_q is None:
            # Time to guess
            guess, confidence_score = game.get_best_guess()
            
            click.echo(f"🤔 My guess: **{guess}** (confidence: {confidence_score:.2f})")
            
            while True:
                result = click.prompt("Am I correct? (yes/no)", type=str).lower()
                if result in ['yes', 'y', 'correct', 'right']:
                    click.echo("🎉 I win! Thanks for playing!")
                    game.update_weights(guess, correct=True)
                    return
                elif result in ['no', 'n', 'wrong', 'incorrect']:
                    actual = click.prompt("What was the correct answer?", type=str)
                    if actual.lower() in [obj.lower() for obj in object_list]:
                        game.update_weights(actual, correct=False)
                        click.echo(f"Thanks! I'll learn from this. The answer was: {actual}")
                    else:
                        click.echo("I don't know that object. I'll try to learn anyway.")
                    return
                else:
                    click.echo("Please answer 'yes' or 'no'")
        
        # Ask question
        question_text = game.question_manager.get_question_text(next_q)
        click.echo(f"Question {game.question_count + 1}: {question_text}")
        
        while True:
            answer = click.prompt("Answer", type=str).lower()
            
            if answer in ['quit', 'exit', 'q']:
                click.echo("Thanks for playing!")
                return
            elif answer == 'help':
                click.echo(game.question_manager.get_answer_help())
                continue
            elif game.question_manager.validate_answer(answer):
                game.submit_answer(next_q, answer)
                break
            else:
                click.echo("Invalid answer. Type 'help' for available options.")


@cli.command()
@click.option('--objects', '-o', type=click.Path(exists=True), required=True,
              help='Path to objects JSON file')
@click.option('--questions', '-q', type=click.Path(exists=True), required=True,
              help='Path to questions JSON file')
@click.option('--output', '-out', type=click.Path(), required=True,
              help='Output path for weights file')
@click.option('--cohort', '-c', type=str,
              help='Demographic cohort ID')
@click.option('--format', '-f', type=click.Choice(['json', 'bin']), default='json',
              help='Output format')
def init_weights(
    objects: str,
    questions: str,
    output: str,
    cohort: Optional[str],
    format: str
) -> None:
    """Initialize random weights for a new game setup."""
    
    # Load data
    with open(objects, 'r') as f:
        objects_data = json.load(f)
    object_list = objects_data.get('objects', objects_data)
    
    question_manager = QuestionManager.from_json(Path(questions))
    
    # Initialize random weights
    n_obj = len(object_list)
    n_q = len(question_manager.questions)
    
    weights = np.random.uniform(-1, 1, (n_obj, n_q))
    
    # Save weights
    output_path = Path(output)
    if format == 'bin':
        output_path = output_path.with_suffix('.bin')
    else:
        output_path = output_path.with_suffix('.json')
    
    WeightIO.save_weights(weights, output_path)
    
    click.echo(f"Initialized weights: {n_obj} objects × {n_q} questions")
    click.echo(f"Saved to: {output_path}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input weights file')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output weights file')
def convert_weights(input: str, output: str) -> None:
    """Convert weights between JSON and binary formats."""
    
    try:
        WeightIO.convert_format(input, output)
        
        # Show compression ratio if converting to binary
        if Path(output).suffix.lower() == '.bin':
            ratio = WeightIO.get_compression_ratio(input, output)
            click.echo(f"Compression ratio: {ratio:.2f}x")
        
        click.echo(f"Converted {input} to {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.option('--data-dir', '-d', type=click.Path(), default='data',
              help='Data directory path')
def setup(data_dir: str) -> None:
    """Set up default data files and directory structure."""
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create default objects
    objects_file = data_path / 'objects.json'
    if not objects_file.exists():
        default_objects = {
            "objects": [
                "Dog", "Cat", "Car", "Apple", "Lion", "Airplane", "Computer",
                "Guitar", "Pizza", "Ocean", "Mountain", "Book", "Phone", "Tree",
                "House", "Bicycle", "Coffee", "Moon", "Fire", "Ice", "Elephant",
                "Butterfly", "Robot", "Cheese", "Diamond", "Hammer", "Flower",
                "Cloud", "Snake", "Painting", "Boat", "Lamp", "Keyboard", "Shoe",
                "Window", "Bridge", "Volcano", "Penguin", "Chocolate", "Mirror"
            ]
        }
        
        with open(objects_file, 'w') as f:
            json.dump(default_objects, f, indent=2)
        
        click.echo(f"Created default objects: {objects_file}")
    
    # Create default questions
    questions_file = data_path / 'questions.json'
    if not questions_file.exists():
        default_questions = {
            "questions": [
                {"text": "Is it alive?", "category": "biology", "difficulty": 1},
                {"text": "Is it bigger than a car?", "category": "size", "difficulty": 1},
                {"text": "Can you eat it?", "category": "food", "difficulty": 1},
                {"text": "Is it made by humans?", "category": "origin", "difficulty": 2},
                {"text": "Can it fly?", "category": "movement", "difficulty": 2},
                {"text": "Is it found in a house?", "category": "location", "difficulty": 2},
                {"text": "Is it electronic?", "category": "technology", "difficulty": 2},
                {"text": "Is it used for entertainment?", "category": "purpose", "difficulty": 2},
                {"text": "Is it soft?", "category": "texture", "difficulty": 2},
                {"text": "Does it have legs?", "category": "anatomy", "difficulty": 2},
                {"text": "Is it expensive?", "category": "value", "difficulty": 3},
                {"text": "Is it colorful?", "category": "appearance", "difficulty": 2},
                {"text": "Can you hold it in your hand?", "category": "size", "difficulty": 1},
                {"text": "Is it used for transportation?", "category": "transport", "difficulty": 2},
                {"text": "Does it make noise?", "category": "sound", "difficulty": 2},
                {"text": "Is it found in nature?", "category": "location", "difficulty": 2},
                {"text": "Is it transparent?", "category": "appearance", "difficulty": 3},
                {"text": "Can it swim?", "category": "movement", "difficulty": 3},
                {"text": "Is it dangerous?", "category": "safety", "difficulty": 3},
                {"text": "Is it used for work?", "category": "purpose", "difficulty": 3}
            ]
        }
        
        with open(questions_file, 'w') as f:
            json.dump(default_questions, f, indent=2)
        
        click.echo(f"Created default questions: {questions_file}")
    
    # Create cohort directories
    cohort_dir = data_path / 'priors' / 'cohort_weights'
    cohort_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up demographic manager with default cohorts
    demographic_manager = DemographicManager(cohort_dir)
    demographic_manager.create_default_cohorts()
    
    click.echo(f"Set up data directory: {data_path}")
    click.echo("Use 'twentyq play' to start playing!")


@cli.command()
@click.option('--cohort', '-c', type=str,
              help='Show stats for specific cohort')
def stats(cohort: Optional[str]) -> None:
    """Show statistics about the game data."""
    
    demographic_manager = DemographicManager()
    
    if cohort:
        # Show specific cohort stats
        stats = demographic_manager.get_cohort_stats(cohort)
        if stats:
            click.echo(f"Cohort: {stats['name']} ({stats['id']})")
            click.echo(f"Type: {stats['type']}")
            click.echo(f"Description: {stats['description']}")
            click.echo(f"Has weights: {stats['has_weights']}")
            if stats['has_weights']:
                click.echo(f"Weight file size: {stats['weight_file_size']} bytes")
        else:
            click.echo(f"Cohort '{cohort}' not found")
    else:
        # Show general stats
        cohorts = demographic_manager.list_cohorts()
        click.echo(f"Available cohorts: {len(cohorts)}")
        
        for cohort_obj in cohorts:
            stats = demographic_manager.get_cohort_stats(cohort_obj.id)
            click.echo(f"  - {cohort_obj.name} ({cohort_obj.id}): {cohort_obj.type.value}")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
