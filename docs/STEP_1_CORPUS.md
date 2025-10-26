# Step 1: Corpus Creation

The 1st step in creating an LLM is creating a corpus.

We start with gathering as much high quality text as possible.

**GOAL:** To create a massive and diverse dataset containing a wide range of human knowledge.

For the purpose of learning, we'll keep things simple and download a simple corpus. We'll use Tiny Shakespeare, which is small and perfect for learning.

💡 **This dataset we're downloading is already processed** - meaning it's already cleaned. It doesn't have HTML tags or noise. It has periods, commas, special characters, uppercase and lower case letters, spacing and numbers which is what we want!

## Code Preview: `step_1_get_corpus.py`

```python
import requests

def download_corpus():
    # tiny Shakespeare dataset that is already cleaned.
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # download the text
    response = requests.get(url)

    if response.status_code == 200:
        # Save the content to a file
        with open("corpus.txt", "w", encoding="utf-8") as file:
            file.write(response.text)
        text = response.text
        print("File downloaded successfully.")
        print("📊 Corpus size: ", len(text), "characters")
        print(f"👀 Preview of first 200 characters:\n{text[:200]}...")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

download_corpus()
```

## RESOURCES

- 🤗 HuggingFace has a nice dataset that is 44TB big open sourced. [Learn more here](https://huggingface.co/datasets).
- 🛠️ If you're interested in creating your own dataset, you can check out [Argilla](https://argilla.io/).

---

[← Back to Main README](../README.md)

