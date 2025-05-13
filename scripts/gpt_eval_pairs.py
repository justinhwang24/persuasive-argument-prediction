from dotenv import load_dotenv
from config import MODEL, TEMPERATURE
from tqdm import tqdm
import pandas as pd
import openai
import os
import random

# extracts the GPT's choice between reply A and reply B
def extract_choice(response_text):
    response_text = response_text.strip().upper()
    if "A" in response_text and not "B" in response_text[:response_text.find("A")]:
        return "A"
    if "B" in response_text and not "A" in response_text[:response_text.find("B")]:
        return "B"
    for line in response_text.splitlines():
        if line.strip().startswith(("A", "B")):
            return line.strip()[0]
    return "UNKNOWN"

# returns prompt asking GPT to pick between a pair of replies
def build_prompt(post, reply_a, reply_b, mode):
    instructions = {
        "predict-explain": 'Which reply is more persuasive and more likely to change the original poster’s view? \
            Answer with "A" or "B", then explain your reasoning.',
        "explain-predict": 'Explain which reply is more persuasive and why. After your explanation, state your choice: "A" or "B".'
    }
    if mode not in instructions:
        raise ValueError("Invalid mode. Choose 'predict-explain' or 'explain-predict'.")

    return f"""
    A Reddit user posted:
    \"{post}\"

    Two users replied:

    Reply A:
    \"{reply_a}\"

    Reply B:
    \"{reply_b}\"

    {instructions[mode]}
    """

# evaluate persuasiveness of replies using GPT
def gpt_eval(dataframe, mode, max_samples=500):
    df = dataframe.sample(n=max_samples, random_state=42)
    predictions = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running GPT ({mode})"):
        # randomize reply order
        if i % 2 == 0:
            reply_a, reply_b = row["positive"], row["negative"]
            correct_label = "A"
        else:
            reply_a, reply_b = row["negative"], row["positive"]
            correct_label = "B"

        prompt = build_prompt(row["post"], reply_a, reply_b, mode)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )
            reply = response.choices[0].message.content
            model_choice = extract_choice(reply)
        except Exception as e:
            print(f"Error at row {i}: {e}")
            model_choice = "ERROR"
            reply = ""

        predictions.append({
            "prompt_mode": mode,
            "post": row["post"],
            "reply_a": reply_a,
            "reply_b": reply_b,
            "correct_label": correct_label,
            "gpt_choice": model_choice,
            "gpt_response": reply
        })

    return pd.DataFrame(predictions)

def main():
    load_dotenv()
    global client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    df_pairs = pd.read_csv("data/argument_pairs.csv")

    results_predict_explain = gpt_eval(df_pairs, mode="predict-explain")
    results_explain_predict = gpt_eval(df_pairs, mode="explain-predict")

    results_predict_explain.to_csv("data/results_predict_then_explain.csv", index=False)
    results_explain_predict.to_csv("data/results_explain_then_predict.csv", index=False)

    for df, label in [(results_predict_explain, "Predict-Then-Explain"), (results_explain_predict, "Explain-Then-Predict")]:
        valid = df[df["gpt_choice"].isin(["A", "B"])]
        acc = (valid["gpt_choice"] == valid["correct_label"]).mean()
        print(f"{label} Accuracy: {acc:.2%} on {len(valid)} samples")

main()