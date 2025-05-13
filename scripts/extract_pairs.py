from config import DATA_PATH
import json
import bz2
import pandas as pd

def extract_usable_pairs(path, sample_size=500):
    pairs = []

    with bz2.open(path, 'rt') as f:
        for line in f:
            try:
                data = json.loads(line)
                post = data['op_text'].strip()

                pos_reply = data['positive']['comments'][0]['body'].strip()
                neg_reply = data['negative']['comments'][0]['body'].strip()

                if not post or not pos_reply or not neg_reply:
                    continue

                pairs.append({
                    'post': post,
                    'positive': pos_reply,
                    'negative': neg_reply
                })

            except (KeyError, IndexError, TypeError, AttributeError):
                continue

            if len(pairs) >= sample_size:
                break

    print(f"Extracted {len(pairs)} usable post–reply pairs")
    return pd.DataFrame(pairs)

def main():
    df = extract_usable_pairs(DATA_PATH)
    df.to_csv("data/argument_pairs.csv", index=False)

main()