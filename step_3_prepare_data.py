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

    def __init__(self, sequence_length=50, train_split=0.9, verbose=True):
        '''
        Parameters:
        sequence_length:
            How many characters the model sees at once (context window)

        train_split:
            fraction to use for training (rest is validation)
            Default: 90% for training, 10% for validation
        
        verbose:
            If True, print progress messages. If False, run silently.
            Default: True
        '''
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.verbose = verbose

        # These will be populated as we execute the methods
        self.char_to_id = None
        self.id_to_char = None
        self.vocab_size = None
        self.encoded_corpus = None
        self.inputs = None
        self.targets = None
        self.train_inputs = None
        self.train_targets = None
        self.validation_inputs = None
        self.validation_targets = None
    
    def load_tokenizer(self, tokenizer_path="tokenizer.json"):
        '''
        Step 1: We will load the tokenizer we created earlier
        Which contains the mappings from characters to numbers and vice versa.
        '''
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Rebuild the dictionaries
        self.char_to_id = tokenizer_data['char_to_id']
        
        # Convert id_to_char keys from strings to integers
        # (JSON saves dictionary keys as strings, but we need them as integers)
        self.id_to_char = {}
        for key, value in tokenizer_data['id_to_char'].items():
            integer_key = int(key)
            self.id_to_char[integer_key] = value
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
        
        if self.verbose:
            print(f"✅ Created {len(self.inputs):,} training examples!")
            print(f"📊 Input shape: {self.inputs.shape}")
            print(f"📊 Target shape: {self.targets.shape}")
            print()
    
    def split_data(self):
        '''
        Step 4: Split the data into training and validation sets.

        Why split the data?
        - TRAINING: The model learns patterns from this data
        - VALIDATION: We test on this to see if the model really learned patterns
        -> This helps us know if the model is truly smart or just memorizing the data
        '''
        total_examples = len(self.inputs)
        split_point = int(self.train_split * total_examples)

        # SPLIT THE DATA AT THAT POINT
        
        # First part: 0-split_point = training
        self.train_inputs = self.inputs[:split_point]
        self.train_targets = self.targets[:split_point]

        # Second part: split_point - end = validation
        self.validation_inputs = self.inputs[split_point:]
        self.validation_targets = self.targets[split_point:]

        if self.verbose:
            print(f"Training examples: {len(self.train_inputs)}")
            print(f"Validation examples: {len(self.validation_inputs)}")

    def save_data(self, output_path="training_data.npz"):
        '''
        STEP 5: Save the training and validation data to a file.
        '''
        # Save as compressed numpy file
        np.savez_compressed(output_path, 
            train_inputs=self.train_inputs,
            train_targets=self.train_targets,
            validation_inputs=self.validation_inputs,
            validation_targets=self.validation_targets)

        if self.verbose:
            print("Data saved to file: training_data.npz")

    def show_summary(self):
        '''
        Display a summary of the data preparation process.
        '''
        if self.verbose:
            print(f"Vocabulary size: {self.vocab_size} characters")
            print(f"Sequence length: {self.sequence_length} characters")
            print(f"Total examples: {len(self.inputs):,}")
            print(f"Training examples: {len(self.train_inputs):,}")
            print(f"Validation examples: {len(self.validation_inputs):,}")
            print()
            print("✅ Data preparation complete!")
            print("File created: - training_data.npz")
            print("Next: Ready to build and train the LLM!")

    def _decode(self, token_ids):
        '''
        Helper method: Convert a list of token IDs back to readable text.
        '''
        text = ""
        for token_id in token_ids:
            text += self.id_to_char[token_id]
        return text
    
    def _show_examples(self, num_examples=3):
        '''
        Display some decoded input-target pairs to see what the model will learn from.
        '''
        print(f"📖 Showing {num_examples} training examples:\n")
        
        for i in range(num_examples):
            input_decoded = self._decode(self.inputs[i])
            target_decoded = self._decode(self.targets[i])
            
            if self.verbose:
                print(f"Example {i+1}:")
                print(f"  Input:  '{input_decoded}'")
                print(f"  Target: '{target_decoded}'")
                print()

    def run_full_pipeline(self):
        '''
        Execute the full data preparation pipeline.
        '''
        self.load_tokenizer() # step 1: load the tokenized data we created from from the tokenizer.json
        self.encode_corpus() # step 2: tokenize corpus
        self.create_training_examples() # step 3: create training examples
        self.split_data() # step 4: split the data into training and validation sets
        self.save_data() # step 5: save the training and validation data to a file
        self.show_summary() # step 6: show a summary of the data preparation process


if __name__ == "__main__":
   data_prep = LLMDataPreperation(sequence_length=50, train_split=0.9, verbose=True)
   data_prep.run_full_pipeline()
