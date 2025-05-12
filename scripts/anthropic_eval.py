from config import MODEL, TEMPERATURE
from dotenv import load_dotenv
from datasets import load_dataset
import openai
import pandas as pd
import time
import os

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dataset = load_dataset("Anthropic/persuasion", split="train[:50]")
df = pd.DataFrame(dataset)

def get_gpt_score(claim, argument):
    prompt = f"""
You and a human annotator are performing this experiment. 
Your goal is to predict the human annotator's performance as closely as possible.
You will be shown a claim and then an argument related to that claim. 
You must rate your agreement with the claim both before and after reading the argument, on a scale from 
1 (strongly disagree) to 100 (strongly agree). Then, subtract the two ratings and multiply by 0.07 to get a persuasion score.

Claim:
"{claim}"

First, before reading the argument, how much do you agree with this claim?

Now, here is the argument:
"{argument}"

After reading the argument, how much do you agree with the claim?

Now compute:
Persuasion Score = (After - Before) * 0.07

Respond in this format:

Before: <1-100>
After: <1–100>
Score: <(After - Before) * 0.07)>
Explanation: <why the argument changed (or didn’t change) your opinion>
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        reply = response.choices[0].message.content.strip()
        score_line = next((line for line in reply.splitlines() if line.strip().startswith("Score:")), "")
        score = float(score_line.split(":")[1].strip()) if "Score:" in score_line else -1 # round or not round
        return score, reply
    except Exception as e:
        print("Error:", e)
        return -1, ""
    
print("Columns:", df.columns.tolist())

results = []
for i, row in df.iterrows():
    claim_text = row["claim"]
    argument_text = row["argument"]
    human_score = row.get("persuasiveness_metric", None)
    gpt_score, explanation = get_gpt_score(claim_text, argument_text)
    results.append({
        "claim": claim_text,
        "argument": argument_text,
        "human_score": human_score,
        "gpt_score": gpt_score,
        "gpt_explanation": explanation
    })
    print(f"{i}: GPT={gpt_score} vs Human={human_score}")

results_df = pd.DataFrame(results)
results_df.to_csv("../data/gpt_vs_human_persuasion.csv", index=False)

valid = results_df[results_df["gpt_score"] > 0]
correlation = valid["gpt_score"].corr(valid["human_score"])
print(f"Correlation between GPT and human scores: {correlation:.2f}")
