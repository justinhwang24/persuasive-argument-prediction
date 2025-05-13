import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/results_predict_then_explain.csv")
# df = pd.read_csv("data/results_explain_then_predict.csv")
df = df[df["gpt_choice"].isin(["A", "B"])].copy()
df["gpt_correct"] = (df["gpt_choice"] == df["correct_label"]).astype(int)

def extract_features(text):
    words = text.split()
    return {
        "num_words": len(words),
        "num_chars": len(text),
        "avg_word_len": len(text) / max(len(words), 1),
        "num_questions": text.count("?"),
        "num_links": len(re.findall(r"http[s]?://", text)),
        "num_bullets": len(re.findall(r"[-*•]", text)),
        "num_hedges": len(re.findall(r"\b(i think|maybe|perhaps|possibly|in my opinion|i believe)\b", text.lower()))
    }

features_a = df["reply_a"].apply(extract_features).apply(pd.Series)
features_b = df["reply_b"].apply(extract_features).apply(pd.Series)

feature_diff = features_a - features_b
feature_diff = feature_diff.add_prefix("a_minus_b_")

df["gpt_chose_a"] = (df["gpt_choice"] == "A").astype(int)

corr_with_gpt_choice = pd.concat([df, feature_diff], axis=1).corr(numeric_only=True)["gpt_chose_a"].filter(like="a_minus_b_").sort_values(ascending=False)

features_chosen = features_a.copy()
features_unchosen = features_b.copy()
for col in features_a.columns:
    features_chosen[col] = df.apply(lambda row: features_a.loc[row.name, col] if row["gpt_choice"] == "A" else features_b.loc[row.name, col], axis=1)
    features_unchosen[col] = df.apply(lambda row: features_b.loc[row.name, col] if row["gpt_choice"] == "A" else features_a.loc[row.name, col], axis=1)

delta_features = features_chosen - features_unchosen
delta_features = delta_features.add_prefix("delta_")

df_combined = pd.concat([df.reset_index(drop=True), delta_features], axis=1)
corr_with_gpt_correct = df_combined.corr(numeric_only=True)["gpt_correct"].filter(like="delta_").sort_values(ascending=False)

correlation_df = pd.DataFrame({
    "Corr: GPT chose A": corr_with_gpt_choice,
    "Corr: GPT was correct": corr_with_gpt_correct
})

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_df.transpose(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of Reply Features with GPT Choice and Correctness")
plt.tight_layout()
plt.show()
