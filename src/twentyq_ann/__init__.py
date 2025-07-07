"""
TwentyQ-ANN: Production-quality 20 Questions engine based on ANN.
"""

__version__ = "0.1.0"
__author__ = "Kurt"
__email__ = "kurt@example.com"

from .core import TwentyQuestionsANN
from .questions import AnswerType, Question, QuestionManager
from .demographics import Cohort, DemographicManager

__all__ = [
    "TwentyQuestionsANN",
    "AnswerType", 
    "Question",
    "QuestionManager",
    "Cohort",
    "DemographicManager",
]
