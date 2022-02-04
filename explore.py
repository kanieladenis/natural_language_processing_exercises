import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
import nltk
import acquire
import prepare

import warnings
warnings.filterwarnings('ignore')



def category_string():
    '''
    This function takes a dataframe, subsets it by column category, selects lemmatized column, and joins each 
    to one string per category
    '''
    # subset df by category type and and join each row to make one category string
    business_lemmatized = ' '.join(df[df.category == 'business'].lemmatized)
    sports_lemmatized = ' '.join(df[df.category == 'sports'].lemmatized)
    technology_lemmatized = ' '.join(df[df.category == 'technology'].lemmatized)
    entertainment_lemmatized = ' '.join(df[df.category == 'entertainment'].lemmatized)
    all_lemmatized = ' '.join(df.lemmatized)
    
    return business_lemmatized, sports_lemmatized, technology_lemmatized, entertainment_lemmatized, all_lemmatized


def category_series():
    '''
    This function take the string, splits it, creates a Series, and finds value counts for each series
    '''
    business_freq = pd.Series(business_lemmatized.split()).value_counts()
    sports_freq = pd.Series(sports_lemmatized.split()).value_counts()
    technology_freq = pd.Series(technology_lemmatized.split()).value_counts()
    entertainment_freq = pd.Series(entertainment_lemmatized.split()).value_counts()
    all_freq = pd.Series(all_lemmatized.split()).value_counts()
    
    return business_freq, sports_freq, technology_freq, entertainment_freq, all_freq


def word_counts():
    '''
    This function concats the value counts sereis into a dataframe 
    '''
    word_counts = pd.concat([business_freq, sports_freq, technology_freq, entertainment_freq, all_freq], axis=1).fillna(0).astype(int)
    word_counts.columns=['business','sports','technology','entertainment','all']
    
    return word_counts


def viz_top_20():
    '''
    This function takes the word_counts value and makes bar graph of the top 20 words used
    '''
    #
    plt.rc('font', size=18)
    word_counts.sort_values('all', ascending=False).head(20)[['business','sports','technology','entertainment']].plot.barh()
    plt.title('Business vs Sports vs Technology vs Entertainment count for the top 20 most frequent words')
    
    
def viz_top_20_percent():
    '''
    '''
    plt.figure(figsize=(16, 9))
    plt.rc('font', size=16)

    (word_counts.sort_values('all', ascending=False)
     .head(20)
     .apply(lambda row: row/row['all'], axis = 1)
     .drop(columns = ['all'])
     .sort_values(by = 'technology')
     .plot.barh(stacked = True, width = 1, ec = 'k')
    )
    plt.title('% of spam vs ham for the most common 20 words')
    
    
    
def biz_bigram():
    '''
    '''
    pd.Series(nltk.bigrams(all_lemmatized.split())).value_counts().head(10).plot.barh()
    plt.title('Top 10 most common spam bigrams')