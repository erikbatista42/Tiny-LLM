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