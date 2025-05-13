import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df_pred = pd.read_csv("data/results_predict_then_explain.csv")
df_expl = pd.read_csv("data/results_explain_then_predict.csv")

cm_pred = confusion_matrix(df_pred["correct_label"], df_pred["gpt_choice"], labels=["A", "B"])
cm_expl = confusion_matrix(df_expl["correct_label"], df_expl["gpt_choice"], labels=["A", "B"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=["A", "B"])
disp_pred.plot(ax=axes[0], cmap="Blues", values_format='d')
axes[0].set_title("Predict-Then-Explain")

disp_expl = ConfusionMatrixDisplay(confusion_matrix=cm_expl, display_labels=["A", "B"])
disp_expl.plot(ax=axes[1], cmap="Greens", values_format='d')
axes[1].set_title("Explain-Then-Predict")

plt.suptitle("GPT Confusion Matrices by Prompt Strategy")
plt.tight_layout()
plt.show()