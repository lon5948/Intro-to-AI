from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from string import digits
from string import punctuation
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''    
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


def lemmatization(text: str) -> str:
    tokens = word_tokenize(text)
    wl = WordNetLemmatizer()
    lemma_tokens= []
    for token in text.split():
        word1 = wl.lemmatize(token,pos = "n")
        word2 = wl.lemmatize(word1,pos = "v")
        word3 = wl.lemmatize(word2,pos = "a")
        word4 = wl.lemmatize(word3,pos = "r")
        lemma_tokens.append(word4)
        
    preprocessed_text = ' '.join(lemma_tokens)
    return preprocessed_text
    

def remove_punctuation(text: str) -> str:
    text = text.replace('<br /><br />',' ')
    tokens = [token for token in text if token not in punctuation]
    preprocessed_text = ''.join(tokens)
    return preprocessed_text


def remove_digit(text: str) -> str:
    tokens = [token for token in text if token not in digits]
    preprocessed_text = ''.join(tokens)
    return preprocessed_text
    
    
def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    preprocessed_text = remove_punctuation(preprocessed_text)
    preprocessed_text = lemmatization(text)
    preprocessed_text = remove_digit(preprocessed_text)
    return preprocessed_text
