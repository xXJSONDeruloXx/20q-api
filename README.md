# 20Q-ANN: Production-Quality 20 Questions Engine

A sophisticated, production-ready implementation of the classic "20 Questions" game using an Artificial Neural Network (ANN) approach based on a target-object × question weight matrix.

## 🎯 Features

- **Advanced ANN Architecture**: Uses a learned weight matrix to intelligently select questions and rank possible answers
- **Multi-valued Answers**: Supports 12 different answer types (yes, no, unknown, sometimes, depends, etc.)
- **Demographic Segmentation**: Cohort-based weight matrices for different user populations
- **Adaptive Learning**: Hebbian-style learning that improves with each game
- **Quantized Storage**: Efficient 8-bit quantized weight storage with 4-10x compression
- **Multiple Interfaces**: Both CLI and REST API interfaces
- **Smart Question Selection**: Two strategies for optimal question selection
- **Confidence-based Guessing**: Configurable confidence thresholds for intelligent guessing

## 🚀 Installation

### Quick Setup (Recommended)

```bash
# Clone or download the project
# cd into the 20q directory

# Install dependencies
pip install -r requirements.txt

# Set up the game
python run_cli.py setup

# Start playing!
python run_cli.py play
```

### Using Poetry (Advanced)

```bash
# Install with Poetry
poetry install

# For API features
poetry install --extras api
```

## 🎮 Quick Start

### CLI Interface

```bash
# Set up default data files
python run_cli.py setup

# Start playing
python run_cli.py play

# Play with specific data files
python run_cli.py play --objects data/objects.json --questions data/questions.json

# Play with a specific demographic cohort
python run_cli.py play --cohort us --confidence 1.5
```

### API Interface

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn

# Start the API server
python run_api.py

# The API will be available at http://localhost:8000
# Visit http://localhost:8000/docs for interactive documentation
```

### Legacy Interface

```bash
# Run the original game (now powered by advanced system)
python 20q.py
```

### Python API

```python
from twentyq_ann import TwentyQuestionsANN, QuestionManager
from twentyq_ann.questions import Question

# Create questions
questions = [
    Question("Is it alive?", "biology"),
    Question("Is it bigger than a car?", "size"),
    Question("Can you eat it?", "food"),
]

# Create game
game = TwentyQuestionsANN(
    objects=["Dog", "Cat", "Car", "Apple"],
    question_manager=QuestionManager(questions),
    confidence_threshold=1.0
)

# Play programmatically
question_idx = game.get_next_question()
if question_idx is not None:
    question_text = game.question_manager.get_question_text(question_idx)
    print(f"Question: {question_text}")
    
    # Submit answer
    game.submit_answer(question_idx, "yes")
    
    # Get best guess when ready
    if game.get_next_question() is None:
        guess, confidence = game.get_best_guess()
        print(f"My guess: {guess} (confidence: {confidence:.2f})")
```

## 🏗️ Architecture

### Core Components

1. **TwentyQuestionsANN**: Main game engine with neural network logic
2. **QuestionManager**: Handles questions and multi-valued answer mapping
3. **DemographicManager**: Manages cohort-based weight matrices
4. **WeightIO**: Handles serialization with JSON and binary formats

### ANN Matrix Operations

The core uses a `(n_objects × n_questions)` weight matrix with two main modes:

1. **Answer→Object Ranking**: Given answers, rank most likely objects
2. **Object→Question Ranking**: Given top objects, find best discriminating question

### Question Selection Strategies

- **Balanced Margin**: Finds questions that split candidates most evenly
- **Confirm Top**: Finds questions that best distinguish the top candidate

## 📊 Multi-valued Answers

The system supports rich answer types beyond simple yes/no:

| Answer | Weight | Description |
|--------|--------|-------------|
| yes | 1.0 | Definitely true |
| no | -1.0 | Definitely false |
| unknown | 0.0 | Don't know |
| sometimes | 0.5 | Sometimes true |
| usually | 0.8 | Usually true |
| rarely | -0.8 | Rarely true |
| maybe | 0.3 | Possibly true |
| probably | 0.7 | Probably true |
| probably_not | -0.7 | Probably false |
| depends | 0.0 | Context-dependent |
| irrelevant | 0.0 | Question doesn't apply |

## 🌍 Demographic Segmentation

Support for different user populations:

```bash
# List available cohorts
poetry run twentyq stats

# Play with a specific cohort
poetry run twentyq play --cohort us

# Create custom cohort
poetry run twentyq init-weights --objects data/objects.json --questions data/questions.json --output data/priors/cohort_weights/custom.json
```

## 💾 Storage Formats

### JSON Format (Human-readable)
```json
{
  "weights": [[0.1, -0.2], [0.3, 0.4]],
  "shape": [2, 2],
  "dtype": "float32"
}
```

### Binary Format (Compressed)
- 8-bit quantized weights
- 4-10x compression ratio
- Optimized for production deployment

```bash
# Convert between formats
poetry run twentyq convert-weights --input weights.json --output weights.bin
```

## 🔧 Configuration

### Game Parameters

- `confidence_threshold`: Minimum score gap for confident guessing (default: 1.0)
- `learning_rate`: Rate of weight updates (default: 0.1)
- `max_questions`: Maximum questions per game (default: 20)

### Question Selection

- `strategy`: "balanced_margin" or "confirm_top"
- Configurable per game session

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/twentyq_ann --cov-report=html

# Run specific test categories
poetry run pytest tests/test_core.py
poetry run pytest tests/test_quantization.py
```

## 📈 Performance

- **Memory**: ~1-10MB for typical datasets (50 objects, 30 questions)
- **Speed**: <1ms per question/answer cycle
- **Compression**: 4-10x reduction with binary format
- **Accuracy**: Improves with play through adaptive learning

## 🔌 API Reference

### REST Endpoints

- `POST /game/start`: Start new game session
- `POST /game/answer`: Submit answer to current question
- `GET /game/state/{session_id}`: Get current game state
- `POST /game/guess`: Submit guess result for learning
- `GET /cohorts`: List available cohorts
- `GET /health`: Health check

### CLI Commands

- `twentyq play`: Start interactive game
- `twentyq setup`: Initialize data files
- `twentyq init-weights`: Create weight matrix
- `twentyq convert-weights`: Convert between formats
- `twentyq stats`: Show statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and install
git clone https://github.com/yourusername/20q-ann.git
cd 20q-ann
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Run linting
poetry run flake8 src/twentyq_ann
poetry run black src/twentyq_ann
poetry run isort src/twentyq_ann
poetry run mypy src/twentyq_ann
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the classic 20 Questions game
- Built with modern Python best practices
- Uses advanced ML techniques for intelligent question selection

## 🔮 Future Enhancements

- [ ] Deep learning integration
- [ ] Multi-language support
- [ ] Real-time multiplayer
- [ ] Advanced analytics dashboard
- [ ] Custom object/question editors
- [ ] Integration with knowledge graphs
- [ ] Voice interface support
- [ ] Mobile app development

---

**Happy Questioning!** 🎭
