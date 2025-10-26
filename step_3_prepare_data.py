import json
import numpy as np

class LLMDataPreperation:
    """
    A class that handles all data preparation steps for training an LLM.
    
    This class takes raw text and prepares it into training-ready format by:
    1. Loading the tokenizer (character-to-number mappings)
    2. Encoding the text corpus into numbers
    3. Creating input-output sequence pairs
    4. Splitting into training and validation sets
    5. Saving the prepared data
    """

    def __init__(self, sequence_length=50, train_split=0.9):
        '''
        Parameters:
        sequence_length:
            How many characters the model sees at once (context window)

        train_split:
            fraction to use for training (rest is validation)
            Default: 90% for training, 10% for validation
        '''
        self.sequence_length = sequence_length
        self.train_split = train_split

        # These will be populated as we execute the methods
        self.char_to_id = None
        self.id_to_char = None
        self.vocab_size = None
        self.encoded_corpus = None
        self.inputs = None
        self.targets = None
        self.train_inputs = None
        self.train_targets = None
        self.val_inputs = None
        self.val_targets = None
    
    def load_tokenizer(self, tokenizer_path="tokenizer.json"):
        '''
        We will load the tokenizer we created earlier which contains the mappings from characters to numbers and vice versa.
        '''
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Rebuild the dictionaries
        self.char_to_id = tokenizer_data['char_to_id']
        self.id_to_char = {int(k): v for k, v in tokenizer_data['id_to_char'].items()}
        self.vocab_size = tokenizer_data['vocab_size']
    
    def encode_corpus(self, corpus_path="corpus.txt"):
        '''
        This step takes all of our corpus text and converts each character to its corresponding token ID.
        '''

        # Load the text file
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Convert each character to its token ID
        self.encoded_corpus = []
        for char in text:
            token_id = self.char_to_id[char]
            self.encoded_corpus.append(token_id)