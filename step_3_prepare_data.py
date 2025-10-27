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
        Step 1: We will load the tokenizer we created earlier
        Which contains the mappings from characters to numbers and vice versa.
        '''
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Rebuild the dictionaries
        self.char_to_id = tokenizer_data['char_to_id']
        self.id_to_char = {int(k): v for k, v in tokenizer_data['id_to_char'].items()}
        self.vocab_size = tokenizer_data['vocab_size']
    
    def encode_corpus(self, corpus_path="corpus.txt"):
        '''
        Step 2: Get all of our corpus text and converts each character to its corresponding token ID.
        '''

        # Load the text file
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Convert each character to its token ID
        self.encoded_corpus = []
        for char in text:
            token_id = self.char_to_id[char]
            self.encoded_corpus.append(token_id)

    def create_training_examples(self):
        '''
        Step 3: Create input-output sequence pairs for training.

        We slide a window across the text to create many training examples.
        Each example shows the model some text (input) and what should come next (target).

        Example with window size 3 on "Hello world":
            Window 1: "Hel" → predict "ell"
            Window 2: "ell" → predict "llo"
            Window 3: "llo" → predict "lo "
            etc.
        '''
        # Create input-output pairs
        inputs_list = [] # What the model sees
        target_list = [] # What we want it to predict

        # Slide a window acros the encoded corpus, one position at a time.
        for position in range(len(self.encoded_corpus) - self.sequence_length):
            # Input: Get N(50?) characters at the starting position
            start = position
            end = position + self.sequence_length
            input_sequence = self.encoded_corpus[start:end]

            # Target: same thing but shifted by 1 character to the right
            target_start = position + 1
            target_end = position + self.sequence_length + 1
            target_sequence = self.encoded_corpus[target_start:target_end]

            # Save this training pair
            inputs_list.append(input_sequence)
            target_list.append(target_sequence)

        # Converts list to numpy arrays for more efficient training
        self.inputs = np.array(inputs_list)
        self.targets = np.array(target_list)
            
        print(f"✅ Created {len(self.inputs):,} training examples!")
        print(f"📊 Input shape: {self.inputs.shape}")
        print(f"📊 Target shape: {self.targets.shape}")
        print()



if __name__ == "__main__":
    data_prep = LLMDataPreperation()
    data_prep.load_tokenizer()
    data_prep.encode_corpus()
    data_prep.create_training_examples()
    print(data_prep.inputs)