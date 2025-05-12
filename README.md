# Persuasion Argument Prediction

This project analyzes GPT-4.1's ability to predict persuasive arguments using data from the Reddit community r/ChangeMyView (CMV). It is inspired by this [paper](https://chenhaot.com/pubs/winning-arguments.pdf) from the Chicago Human+AI Lab which used logistic regression.
## Structure

- `scripts/`: All evaluation scripts
- `data/`: Preprocessed datasets and GPT outputs
- `config.py`: Set the model and temperature
- `report.md`: Final write-up

## Prompt Modes

- **Predict-Then-Explain**  
- **Explain-Then-Predict**

## Results Summary (GPT-4.1)
- Predict-Then-Explain: 58%
- Explain-Then-Predict: **66%**

## How to Run

1. Install dependencies: `pip install openai pandas tqdm python-dotenv`
2. Add your API key in a `.env` file:
