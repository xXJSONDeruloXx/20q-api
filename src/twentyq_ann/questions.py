"""
Question management and multi-valued answer mapping.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path


class AnswerType(Enum):
    """Enumeration of possible answer types with their semantic meanings."""
    
    # Core binary answers
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"
    
    # Conditional answers
    SOMETIMES = "sometimes"
    DEPENDS = "depends"
    USUALLY = "usually"
    RARELY = "rarely"
    
    # Qualitative answers
    MAYBE = "maybe"
    PROBABLY = "probably"
    PROBABLY_NOT = "probably_not"
    
    # Meta answers
    IRRELEVANT = "irrelevant"
    
    @classmethod
    def from_string(cls, answer: str) -> 'AnswerType':
        """Convert string answer to AnswerType."""
        answer_lower = answer.lower().strip()
        
        # Handle common variations
        variations = {
            'y': cls.YES,
            'n': cls.NO,
            'dunno': cls.UNKNOWN,
            'dont_know': cls.UNKNOWN,
            "don't_know": cls.UNKNOWN,
            'idk': cls.UNKNOWN,
            'not_sure': cls.UNKNOWN,
            'sort_of': cls.SOMETIMES,
            'kind_of': cls.SOMETIMES,
            'it_depends': cls.DEPENDS,
            'most_of_the_time': cls.USUALLY,
            'not_often': cls.RARELY,
            'not_really': cls.PROBABLY_NOT,
            'nah': cls.PROBABLY_NOT,
            'irrelevant': cls.IRRELEVANT,
            'not_applicable': cls.IRRELEVANT,
            'na': cls.IRRELEVANT,
        }
        
        if answer_lower in variations:
            return variations[answer_lower]
        
        # Try direct enum match
        for answer_type in cls:
            if answer_type.value == answer_lower:
                return answer_type
        
        # Default to unknown for unrecognized answers
        return cls.UNKNOWN


@dataclass
class Question:
    """Represents a question with metadata."""
    
    text: str
    category: str = "general"
    difficulty: int = 1  # 1-5 scale
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class QuestionManager:
    """Manages questions and answer weight mapping."""
    
    # Default weight mapping for answer types
    DEFAULT_WEIGHTS = {
        AnswerType.YES: 1.0,
        AnswerType.NO: -1.0,
        AnswerType.UNKNOWN: 0.0,
        AnswerType.SOMETIMES: 0.5,
        AnswerType.DEPENDS: 0.0,
        AnswerType.USUALLY: 0.8,
        AnswerType.RARELY: -0.8,
        AnswerType.MAYBE: 0.3,
        AnswerType.PROBABLY: 0.7,
        AnswerType.PROBABLY_NOT: -0.7,
        AnswerType.IRRELEVANT: 0.0,
    }
    
    def __init__(
        self, 
        questions: Optional[List[Question]] = None,
        answer_weights: Optional[Dict[AnswerType, float]] = None
    ):
        """
        Initialize question manager.
        
        Args:
            questions: List of Question objects
            answer_weights: Custom mapping of answer types to weights
        """
        self.questions = questions or []
        self.answer_weights = answer_weights or self.DEFAULT_WEIGHTS.copy()
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'QuestionManager':
        """Load questions from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for q_data in data.get('questions', []):
            if isinstance(q_data, str):
                # Simple string format
                questions.append(Question(text=q_data))
            else:
                # Full object format
                questions.append(Question(
                    text=q_data['text'],
                    category=q_data.get('category', 'general'),
                    difficulty=q_data.get('difficulty', 1),
                    tags=q_data.get('tags', [])
                ))
        
        # Load custom answer weights if provided
        answer_weights = None
        if 'answer_weights' in data:
            answer_weights = {}
            for answer_str, weight in data['answer_weights'].items():
                try:
                    answer_type = AnswerType.from_string(answer_str)
                    answer_weights[answer_type] = float(weight)
                except Exception:
                    pass  # Skip invalid entries
        
        return cls(questions=questions, answer_weights=answer_weights)
    
    def to_json(self, filepath: Path) -> None:
        """Save questions to JSON file."""
        data = {
            'questions': [
                {
                    'text': q.text,
                    'category': q.category,
                    'difficulty': q.difficulty,
                    'tags': q.tags
                }
                for q in self.questions
            ],
            'answer_weights': {
                answer_type.value: weight
                for answer_type, weight in self.answer_weights.items()
            }
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_question(self, question: Question) -> None:
        """Add a new question."""
        self.questions.append(question)
    
    def remove_question(self, index: int) -> None:
        """Remove a question by index."""
        if 0 <= index < len(self.questions):
            del self.questions[index]
    
    def get_question_text(self, index: int) -> str:
        """Get question text by index."""
        if 0 <= index < len(self.questions):
            return self.questions[index].text
        return ""
    
    def get_answer_weight(self, answer_type: AnswerType) -> float:
        """Get the weight for an answer type."""
        return self.answer_weights.get(answer_type, 0.0)
    
    def set_answer_weight(self, answer_type: AnswerType, weight: float) -> None:
        """Set the weight for an answer type."""
        self.answer_weights[answer_type] = weight
    
    def get_questions_by_category(self, category: str) -> List[int]:
        """Get question indices by category."""
        return [
            i for i, q in enumerate(self.questions)
            if q.category == category
        ]
    
    def get_questions_by_difficulty(self, difficulty: int) -> List[int]:
        """Get question indices by difficulty level."""
        return [
            i for i, q in enumerate(self.questions)
            if q.difficulty == difficulty
        ]
    
    def get_questions_by_tag(self, tag: str) -> List[int]:
        """Get question indices by tag."""
        return [
            i for i, q in enumerate(self.questions)
            if tag in q.tags
        ]
    
    def get_available_answers(self) -> List[str]:
        """Get list of available answer strings."""
        return [answer_type.value for answer_type in AnswerType]
    
    def validate_answer(self, answer: str) -> bool:
        """Validate if an answer string is recognized."""
        try:
            AnswerType.from_string(answer)
            return True
        except Exception:
            return False
    
    def get_answer_help(self) -> str:
        """Get help text explaining available answers."""
        help_text = "Available answers:\\n"
        help_text += "• yes/y - Definitely yes\\n"
        help_text += "• no/n - Definitely no\\n"
        help_text += "• unknown/dunno/idk - Don't know\\n"
        help_text += "• sometimes/sort_of - Sometimes true\\n"
        help_text += "• usually - Usually true\\n"
        help_text += "• rarely - Rarely true\\n"
        help_text += "• maybe - Possibly true\\n"
        help_text += "• probably - Probably true\\n"
        help_text += "• probably_not - Probably false\\n"
        help_text += "• depends/it_depends - Depends on context\\n"
        help_text += "• irrelevant - Question doesn't apply\\n"
        return help_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the question set."""
        categories = {}
        difficulties = {}
        
        for question in self.questions:
            # Count categories
            categories[question.category] = categories.get(question.category, 0) + 1
            
            # Count difficulties
            difficulties[question.difficulty] = difficulties.get(question.difficulty, 0) + 1
        
        return {
            'total_questions': len(self.questions),
            'categories': categories,
            'difficulties': difficulties,
            'answer_types': len(self.answer_weights)
        }
