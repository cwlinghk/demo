'''
Illustrate the calculations on the discounts D in KenLM (for ngram = 3)
Kenneth Heafield, Scalable Modified Kneser-Ney Language Model Estimation Kenneth, 2013
Hui Zhang and David Chiang, Kneser-Ney Smoothing on Expected Counts, 2014
'''

import math
import nltk
import codecs
from collections import Counter

def count_ngram(n,sents):
    ngrams = [[i for i in nltk.ngrams(sent,n)] for sent in sents]
    ngrams = [j for i in ngrams for j in i] #flatten
    ngrams_count = nltk.FreqDist(ngrams)
    return ngrams, ngrams_count

def cnt_distinct(nPlus1_gram_f, ngram):
    '''
    Count distinguishable n+1grams whose last n tokens == ngram
    eg. |{(w, w1, w2, ..., wn): count((w, w1, w2, ..., wn))> 0}|, where (w1, w2, ..., wn) == ngram
    '''
    cnt = 0
    len_ngram = len(ngram)
    for k in nPlus1_gram_f.keys():
        k_list = list(k)
        if k_list[-len_ngram:] == list(ngram):
            cnt+=1
    return cnt

def cal_d(ncount_freq):
    '''
    Hui Zhang and David Chiang, Kneser-Ney Smoothing on Expected Counts, 2014
    Eq. 13
    '''
    n1 = ncount_freq[1] #number of items with frequenc ==1
    n2 = ncount_freq[2] #number of items with frequenc ==2
    n3 = ncount_freq[3]
    n4 = ncount_freq[4]
    
    y=n1/(n1+2*n2)
    d1=1-2*y*n2/n1
    d2=2-3*y*n3/n2
    d3=3-4*y*n4/n3
    
    return d1, d2, d3

def load_exported(fpath):
    sents=[]
    with codecs.open(fpath, 'r', encoding='utf8') as f:
        raw = f.read()
        for sent in raw.split('\n'):
            sent=sent.split()
            if sent == []: continue
            sents += [['<s>']+ sent +['</s>']]
    return sents

def export_tokenized_sentences(fraw, ftokens):
    #tokenize
    with codecs.open(fraw, 'r', encoding='utf8') as f:
        txt = f.read()
        txt_sents = nltk.sent_tokenize(txt)
        txt_tokens = [nltk.word_tokenize(sent) for sent in txt_sents]
    #export
    with codecs.open(ftokens, 'w+', encoding='utf8') as f:
        for sent in txt_tokens:
            f.write(' '.join(sent) +'\n')
            
if __name__ == '__main__':
    #file paths
    fraw = r"test_corpus_raw.txt" #raw text
    ftokens = r"test_corpus_tokens.txt" #tokenized sentences
    
    #generate and load tokenized sentences
    export_tokenized_sentences(fraw, ftokens) #ftokens is for kenlm to build arpa
    
    #load tokenized sentences
    sents = load_exported(ftokens) #add <s> and </s> to each sentence
    
    #extract and count ngram
    _, gram1_f = count_ngram(1,sents) #unigram count
    _, gram2_f = count_ngram(2,sents) #bigram count
    _, gram3_f = count_ngram(3,sents) #trigram count
    
    #calculation of discounts
    #section 3.2, Scalable Modified Kneser-Ney Language Model Estimation Kenneth
    #D1
    freq_dict= {unigram:cnt_distinct(gram2_f,unigram) for unigram in gram1_f.keys()} #number of distinct types
    ncount_freq = Counter(freq_dict.values()) #{freq: freq of freq}
    d1, d2, d3 = cal_d(ncount_freq)
    print ('D_n(k)\t D_11, D_12, D_13: %f, %f, %f'%(d1,d2,d3))
    
    #D2 (bigram discount need trigram data)
    freq_dict= {bigram:cnt_distinct(gram3_f,bigram) if bigram[0]!='<s>' else gram2_f[bigram] for bigram in gram2_f.keys()} 
    ncount_freq = Counter(freq_dict.values())
    d1, d2, d3 = cal_d(ncount_freq)
    print ('D_n(k)\t D_21, D_22, D_23: %f, %f, %f'%(d1,d2,d3))
    
    #D3
    ncount_freq = Counter(gram3_f.values()) #number of trigram count
    d1, d2, d3 = cal_d(ncount_freq)
    print ('D_n(k)\t D_31, D_32, D_33: %f, %f, %f'%(d1,d2,d3))
