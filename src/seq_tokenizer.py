# Make sure to install the following before running:
# conda activate ukbio1.0; pip install --user --upgrade git+https://github.com/huggingface/transformers.git
from transformers import AutoTokenizer

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-1000g")

# Create a dummy dna sequence and tokenize it
# replaces these sequences with those stored in 'neuroMRI_qced_seqs.csv'
tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt")["input_ids"]

def tokenize_seqs(self):
    # Import the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-1000g")

    # Create a dummy dna sequence and tokenize it
    # replaces these sequences with those stored in 'neuroMRI_qced_seqs.csv'
    tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt")["input_ids"]