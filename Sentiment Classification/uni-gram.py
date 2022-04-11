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
    def __init__(self, config, n=1):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config


    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)


    def get_ngram(self, corpus_tokenize: List[List[str]]):
        count = 0
        uni_dict = {}
        for corpus in corpus_tokenize:
            for index in range(len(corpus)):
                if corpus[index] in uni_dict:
                    uni_dict[corpus[index]] += 1
                else:
                    uni_dict[corpus[index]] = 1  
                count += 1
        
        prob_dict = {}
        for uniword,uni_num in uni_dict.items():
            if uniword in prob_dict:
                prob_dict[uniword][uniword] = uni_num / count
            else:
                prob_dict.update({uniword:{uniword:uni_num / count}})
        return prob_dict,uni_dict

    
    def train(self, df):
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]    
        self.model,self.features = self.get_ngram(corpus)
         

    def compute_perplexity(self, df_test) -> float:
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        entropy = 0
        count = 0
        for sentence in corpus:
            for index in range(len(sentence)):
                if sentence[index] in self.model:
                    prob = self.model[sentence[index]][sentence[index]]
                    entropy += math.log(prob,2)
                    count += 1
        
        entropy = (-1/count) * entropy
        perplexity = math.pow(2,entropy)
        # end your code
        return perplexity


    def train_sentiment(self, df_train, df_test):
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
            for index in range(len(sentence)):
                string_list.append(sentence[index])
            for i,(string,value) in enumerate(Fea):
                if i >= min(feature_num,len(Fea)):
                    break
                train_list[i] = string_list.count(string)
            train_corpus_embedding.append(train_list)
        
        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        for sentence in corpus:
            test_list = [0] * min(feature_num,len(Fea))
            string_list = []
            for index in range(len(sentence)):
                string_list.append(sentence[index])
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
