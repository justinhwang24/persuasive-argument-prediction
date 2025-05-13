# Persuasive Argument Prediction

This project investigates whether large language models (LLMs), specifically GPT-3.5 Turbo, can predict which of two responses in an online discussion is more persuasive. Using the ChangeMyView (CMV) dataset, we prompt GPT to select the reply most likely to earn a “delta” from the original poster, which signals a persuasive impact. We compare two prompting strategies: predict-then-explain, where GPT chooses first and then justifies its answer, and explain-then-predict, where it reasons before committing to a choice. Across 500 randomized A/B reply pairs, the explain-then-predict strategy outperforms predict-then-explain, achieving 58.2% accuracy at temperature 0 compared to 52.8%. We find that a higher temperature at 0.5 narrows the discrepancy, with 58.4% versus 57.0% accuracy, respectively. Feature analysis reveals that GPT relies heavily on surface cues such as length and formatting.

## Key Findings

- **Explanation-first prompting improves accuracy**, especially at temperature = 0.
- GPT-3.5 relies heavily on surface-level features like length and formatting.
- At higher temperatures, GPT’s accuracy improves in PTE due to reduced overconfidence.
- Feature-level analysis shows that model preferences do not always align with what makes an argument persuasive to humans.
