import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import nltk

nltk.download('stopwords')
nltk.download('punkt')

punctuation_irritants = ',.-_&%>'
stopwords = set(stopwords.words('english'))

def clean_sent(sent):
    for irritant in punctuation_irritants:
        sent = sent.replace(irritant, ' ')
    return sent
def tokenize(value, number_required_flag=False):
    overall_value = []
    value = value.lower()
    # For each column tokenize into sentences and words
    for sent in sent_tokenize(value):
        sent_data = []
        # The following line is the fastest way to remove punctuations
        sent = clean_sent(sent)
        # The following line removes digits
        if number_required_flag:
            pass
        else:
            sent = re.sub(r'\d+', ' ', sent)
        for word in word_tokenize(sent):
            if word not in stopwords and word != '' and word.isalnum() and len(word) > 1:
                sent_data.append(word)
            else:
                pass
        overall_value.append(' '.join(sent_data))
    return ' '.join(overall_value).encode('ascii', 'ignore')

main_df = pd.read_csv('000', delimiter='\x01')
main_df.fillna('', inplace=True)
distinct_attributes = main_df['attribute_name'].unique().tolist()
for attribute in distinct_attributes:
    df = main_df[main_df['attribute_name'] == attribute]
    print ("1. Read Data")
    df = df[~df['value_list'].str.contains('<-->')]
print(df.columns)
df['all_data_tokenized'] = df['product_name'].map(tokenize) + ' ' + df['copy'].map(tokenize) + ' ' + df['feature_and_benefits_cleaned']
print(df['all_data_tokenized'])
