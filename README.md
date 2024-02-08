# NLP-A3-Machine-Translation
 
English-to-Thai machine translation app using Transformer Seq2Seq model.

## Dataset Used:

OPUS-100 is an English-centric multilingual corpus randomly sampled from the OPUS collection covering 100 languages. \
For English-Thai language pairs subset, there are 1,000,000 samples for training, 2,000 samples for validation, and 2,000 samples for testing. \
In this project, we reduced the size of training samples to only 20,000 due to computational limitations. \
Link: https://huggingface.co/datasets/opus100

## Word Tokenization

Tokenization of English text was done using the spaCy tokenizer which can handle special character and punctuation very well. \
For Thai text however, we used the newmm, a dictionary-based tokenizer included in PyThaiNLP library.

## 
