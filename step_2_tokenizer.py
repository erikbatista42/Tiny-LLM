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

def tokenize_corpus():
    # STEP 1: Load the corpus
    with open('corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # STEP 2: Create tokenizer and build vocabulary
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocabulary(text)

    # STEP 3: Test the tokenizer
    test_text = "hello world!"


    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    # print(encoded)
    # print(decoded)
    if test_text == decoded:
        print("✅ Perfect match! The tokenizer works!")
    else:
        print("❌ Something went wrong - the tokenizer does not work!")

    # Step 4: Save the tokenizer
    tokenizer.save()

    


if __name__ == "__main__":
    tokenize_corpus()
    


