import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

class Trigram:
    def __init__(self, config, n=3):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config


    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)


    def get_ngram(self, corpus_tokenize: List[List[str]]):
        bi_dict = {}
        tri_dict = {}
        for corpus in corpus_tokenize:
            for index in range(len(corpus)-2):
                string1 = corpus[index] + " " + corpus[index+1]
                if string1 in bi_dict:
                    bi_dict[string1] += 1
                else:
                    bi_dict[string1] = 1  
                string2 = corpus[index] + " " + corpus[index+1] + " " + corpus[index+2]
                if string2 in tri_dict:
                    tri_dict[string2] += 1 
                else:
                    tri_dict[string2] = 1 
        
        prob_dict = {}
        for triword,tri_num in tri_dict.items():
            spl = triword.split()
            string = spl[0] + " " + spl[1]
            if string in prob_dict:
                prob_dict[string][spl[2]] = tri_num / bi_dict[string]
            else:
                prob_dict.update({string:{spl[2]:tri_num / bi_dict[string]}})
        return prob_dict,tri_dict

    
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
            for index in range(len(sentence)-2):
                string = sentence[index] + " " + sentence[index+1]
                if string in self.model and sentence[index+2] in self.model[string]:
                    prob = self.model[string][sentence[index+2]]
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
            for index in range(len(sentence)-2):
                string_list.append(sentence[index] + " " + sentence[index+1]+ " " + sentence[index+2])
            for i,(string,value) in enumerate(Fea):
                if i >= min(feature_num,len(Fea)):
                    break
                train_list[i] = string_list.count(string)
            train_corpus_embedding.append(train_list)
        
        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        for sentence in corpus:
            test_list = [0] * min(feature_num,len(Fea))
            string_list = []
            for index in range(len(sentence)-2):
                string_list.append(sentence[index] + " " + sentence[index+1]+ " " + sentence[index+2])
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