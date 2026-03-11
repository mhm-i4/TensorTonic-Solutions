import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        # special tokens
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        id = 4

        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = id
                    self.id_to_word[id] = word
                    id += 1

        self.vocab_size = len(self.word_to_id)

    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        enc = []

        for word in text.lower().split():
            if word in self.word_to_id:
                enc.append(self.word_to_id[word])
            else:
                enc.append(self.word_to_id[self.unk_token])

        return enc
    

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        result = []

        for i in range(len(ids)):
            if ids[i] in self.id_to_word:
                result.append(self.id_to_word[ids[i]])

        return " ".join(result)