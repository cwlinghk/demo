# KenLM_discounts
KenLM implemented the modified Kneser-Ney smoothing. However, it is often not easy to understand the formulars without simple codecs and toy examples. Thus, this respository illustrates the calculation of the discounts $D$ in KenLM (for ngram = 3).

![alt text](https://github.com/cwlinghk/demo/KenLM_discounts/blob/master/img/Capture.JPG)
Fig. 1. Discounts calculated by KenLM during language model training on the test_corpus_tokens.txt  

# Calculating $D$ from a corpus
The codes first splits and tokenizes the text in 'test_corpus_raw.txt' to 'test_corpus_tokens.txt', which can be the training data for KenLM. Then, it calculates and prints out the discounts $D_n(k)$.

![alt text](https://github.com/cwlinghk/demo/KenLM_discounts/blob/master/img/Capture2.JPG)
Fig. 2. Discounts caluculated by the codes.

# Files
test_corpus_raw.txt------Raw text sentences from Wikipedia.
test_corpus_tokens.txt---Tokenized sentences from test_corpus_raw.txt. It can be used for traning KenLM.
main.py------------------The main codes.
