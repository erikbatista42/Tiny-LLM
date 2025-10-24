# Tiny LLM

## 1. Corpus Creation

The 1st step in creating an LLM is creating a corpus.

We start with gathering as much high quality text as possible.

**GOAL:** To create a massive and diverse dataset containing a wide range of human knowledge.

For the purpose of learning, we'll keep things simple and download a simple corpus. We'll use Tiny Shakespeare, which is small and perfect for learning.

💡 **This dataset we're downloading is already processed** - meaning it's already cleaned. It doesn't have HTML tags or noise. It has periods, commas, special characters, uppercase and lower case letters, spacing and numbers which is what we want!

### Code Preview: `step_1_get_corpus.py`

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

### RESOURCES

- 🤗 HuggingFace has a nice dataset that is 44TB big open sourced. [Learn more here](https://huggingface.co/datasets).
- 🛠️ If you're interested in creating your own dataset, you can check out [Argilla](https://argilla.io/).

---

## 2. Creating a Tokenizer

**Tokenization** - the process of translating text to numbers.

In this step, we convert tokens into numbers. Once the words are transformed into numbers, they are referred to as **tokens**. Imagine tokenization as creating a codebook, such as "A=1, B=2, C=3…".

### Why do we need to do this?

Because we will be giving our tokens to our neural networks to process. **Neural networks only understand numbers.** We will cover neural networks later.

### CREATING A TOKENIZER

```python
import json # This helps us save our tokenizer to a file

class CharacterTokenizer:
    '''
    Think of this class as a toolbox that has different tools for converting text to numbers and vice versa.
    '''

    def __init__(self):
        self.char_to_id = {} # Will store: "a" -> 0, "b" -> 1, etc.
        self.id_to_char = {} # Will store: 0 -> "a", 1 -> "b", etc.
        self.vocab_size = 0  # Will store: How many unique characters

    def build_vocabulary(self, text):
        '''
        What this does:
        1. Finds all unique characters in the text
        2. Assigns each character a unique number (Token ID)
        3. Creates both dictionaries (Char -> ID and ID -> Char)
        '''

        # STEP 1: Find unique characters in text
        unique_chars = set(text)
        unique_chars = sorted(unique_chars)
        self.vocab_size = len(unique_chars)

        # Step 2: Create the char_to_id dictionary
        for index, char in enumerate(unique_chars):
            self.char_to_id[char] = index
        
        # Step 3: Create the id_to_char dictionary
        for char, char_id in self.char_to_id.items():
            self.id_to_char[char_id] = char

    def encode(self, text):
        '''
        Convert text to list of numbers (Token IDs)

        EXAMPLE: "hi" -> [23, 24] -- if 'h'=23 and 'i'=34
        '''
        # Go through each character and look up its number
        token_ids = []
        for char in text:
            token_ids.append(self.char_to_id[char])

        return token_ids

    def decode(self, token_ids):
        '''
        Convert a list of numbers (Token IDs) back into text

        EXAMPLE: [23, 24] -> "hi" -- if 23='h' and 24='i'
        '''
        text = ""
        for token_id in token_ids:
            text += self.id_to_char[token_id]
        
        return text

    def save(self, filename="tokenizer.json"):
        ''' Save our tokenizer to a JSON file so we can use it later.'''
        # Convert id_to_char keys from int to str for JSON
        id_to_char_as_strings = {}
        for key, value in self.id_to_char.items():
            id_to_char_as_strings[str(key)] = value

        data_to_save = {
            'char_to_id': self.char_to_id,
            'id_to_char': id_to_char_as_strings,
            'vocab_size': self.vocab_size
        }

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
        
        print("Tokenizer file saved! You can now use this file for later.")
```

### Tokenization Concepts

#### **Vocabulary (vocab)**
This is the complete list of all unique tokens your tokenizer knows. For example:

```python
vocabulary = [' ', '!', ',', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print(vocabulary)
```

This shows all unique items in the corpus, sorted.

#### **Vocab size**
The count of items in your vocabulary.

```python
# Define the vocabulary list again to ensure it is available for use.
vocabulary = [' ', '!', ',', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Print the length of the vocabulary to get the vocab size.
print(len(vocabulary))  # Output: 56
```

**Why vocab size matters:**
- Affects memory and computation speed
- Determines the range of token IDs

**Examples:**
- **Character-level:** vocab_size ≈ 50-100 (small!)
- **Word-level:** vocab_size ≈ 50,000+ (large!)
- **BPE:** vocab_size ≈ 30,000-50,000 (medium)

#### **Token ID**
The number assigned to each token.

```python
# Mapping from character to ID
char_to_id = {
    'a': 0,
    'b': 1,
    'c': 2,
    ' ': 3
}
# So 'a' has token ID = 0, 'b' has token ID = 1, etc.
```

#### **Encoding**
Converting text to numbers (token IDs) → This is what you feed into neural networks!

```python
text = "hi"
encoded = tokenizer.encode(text)
# Result: [23, 34]  (if 'h'=23, 'i'=34)
```

#### **Decoding**
Converting numbers to text (back to human readable form).

```python
token_ids = [23, 34]
decoded = tokenizer.decode(token_ids)
# Result: "hi"
```

#### **Sequence length**
How many tokens are in your encoded text.

```python
text = "hello"
encoded = [23, 14, 31, 31, 38]
sequence_length = len(encoded)  # 5 tokens
```

---

### MAIN METHODS OF TOKENIZATION

#### **Type 1: Character level (what we're doing)** 
Each character = 1 token

```
"hello" → ['h', 'e', 'l', 'l', 'o'] → [23, 14, 31, 31, 38]
```

**Pros:**
- ✅ Simple to understand
- ✅ Small vocabulary (~65 tokens)
- ✅ Never has "unknown" characters
- ✅ Perfect for learning

**Cons:**
- ❌ Long sequences (slow for large texts)

---

#### **Type 2: Word-Level**
Each word = 1 token

```
"hello world" → ['hello', 'world'] → [5234, 8912]
```

**Pros:**
- ✅ Short sequences
- ✅ Each token has meaning

**Cons:**
- ❌ Huge vocabulary (50,000+ words)
- ❌ Can't handle typos or new words
- ❌ Needs special "unknown" token

---

#### **Type 3: Subword-Level (BPE, WordPiece, etc.)**
Smart chunks = tokens

```
"hello" → ['hel', 'lo'] → [342, 891]
"unhappiness" → ['un', 'happy', 'ness'] → [12, 456, 789]
```

**Pros:**
- ✅ Balanced vocabulary size (~30,000)
- ✅ Shorter sequences than character-level
- ✅ Can handle unknown words by breaking them into parts
- ✅ Used by GPT, BERT, etc.

**Cons:**
- ❌ More complex to implement
- ❌ Harder to understand

---

### RESOURCES

- 🤗 HuggingFace has free lessons on BPE and Subword-Level if interested [[see here](https://huggingface.co/learn/nlp-course/chapter6/5)]
- 🖼️ Visualize tokenization with Tiktokenizer [[see here](https://tiktokenizer.vercel.app/)]

