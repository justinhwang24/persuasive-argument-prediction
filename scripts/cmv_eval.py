from config import MODEL, TEMPERATURE
import pandas as pd
import openai
import os
import time
import random

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df_pairs = pd.read_csv("../data/usable_argument_pairs.csv")

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
        score = float(score_line.split(":")[1].strip()) if "Score:" in score_line else -1.0
        return score, reply
    except Exception as e:
        print("Error:", e)
        return -1.0, ""

results = []
for i, row in df_pairs.iterrows():
    post = row["post"]
    reply_pos = row["positive"]
    reply_neg = row["negative"]

    # randomize order
    if random.random() < 0.5:
        reply_a, reply_b = reply_pos, reply_neg
        correct = "A"
    else:
        reply_a, reply_b = reply_neg, reply_pos
        correct = "B"

    score_a, expl_a = get_gpt_score(post, reply_a)
    score_b, expl_b = get_gpt_score(post, reply_b)

    predicted = "A" if score_a > score_b else "B"
    is_correct = predicted == correct

    results.append({
        "post": post,
        "reply_a": reply_a,
        "reply_b": reply_b,
        "score_a": score_a,
        "score_b": score_b,
        "predicted": predicted,
        "correct": correct,
        "is_correct": is_correct,
        "explanation_a": expl_a,
        "explanation_b": expl_b
    })

    print(f"{i}: A={score_a}, B={score_b} -> predicted={predicted} ({'(O)' if is_correct else '(X)'})")

results_df = pd.DataFrame(results)
results_df.to_csv("../data/cmv_anthropic_style_scoring_results.csv", index=False)

accuracy = results_df["is_correct"].mean()
print(f"\nCMV accuracy using Anthropic-style scoring: {accuracy:.2%}")
