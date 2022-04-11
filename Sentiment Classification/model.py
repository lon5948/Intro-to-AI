import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config


    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)


    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        word_dict = {}
        corpus_dict = {}
        for corpus in corpus_tokenize:
            for index in range(len(corpus)-1):
                if corpus[index] in word_dict:
                    word_dict[corpus[index]] += 1
                else:
                    word_dict[corpus[index]] = 1  
                string = corpus[index] + " " + corpus[index+1]
                if string in corpus_dict:
                    corpus_dict[string] += 1 
                else:
                    corpus_dict[string] = 1 
        
        prob_dict = {}
        for pair,corpus_num in corpus_dict.items():
            spl = pair.split()
            if spl[0] in prob_dict:
                prob_dict[spl[0]][spl[1]] = corpus_num / word_dict[spl[0]]
            else:
                prob_dict.update({spl[0]:{spl[1]:corpus_num / word_dict[spl[0]]}})
        return prob_dict,corpus_dict
        
        # end your code

    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        # [CLS] represents start of sequence
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]    
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model,self.features = self.get_ngram(corpus)
         

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        # begin your code (Part 2)
        entropy = 0
        count = 0
        for sentence in corpus:
            for index in range(len(sentence)-1):
                if sentence[index] in self.model and sentence[index+1] in self.model[sentence[index]]:
                    prob = self.model[sentence[index]][sentence[index+1]]
                    entropy += math.log(prob,2)
                    count += 1
        
        entropy = (-1/count) * entropy
        perplexity = math.pow(2,entropy)
        # end your code
        return perplexity


    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 500
        Fea = sorted(self.features.items(),key = lambda x:x[1],reverse=True)
        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
       
        
        train_corpus_embedding = []
        test_corpus_embedding = []
        
        corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        for sentence in corpus:
            train_list = [0] * min(feature_num,len(Fea))
            string_list = []
            for index in range(len(sentence)-1):
                string_list.append(sentence[index] + " " + sentence[index+1])
            for i,(string,value) in enumerate(Fea):
                if i >= min(feature_num,len(Fea)):
                    break
                train_list[i] = string_list.count(string)
            train_corpus_embedding.append(train_list)
        
        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        for sentence in corpus:
            test_list = [0] * min(feature_num,len(Fea))
            string_list = []
            for index in range(len(sentence)-1):
                string_list.append(sentence[index] + " " + sentence[index+1])
            for i,(string,value) in enumerate(Fea):
                if i >= min(feature_num,len(Fea)):
                    break
                test_list[i] = string_list.count(string)
            test_corpus_embedding.append(test_list)
        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))