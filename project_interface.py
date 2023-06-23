
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import pyterrier as pt
import os, datetime, string, re
from joblib import dump, load
from sklearn.model_selection import train_test_split
import socket

socket.getaddrinfo('localhost', 8080)
# os.environ["JAVA_HOME"] ='C:/Program Files/Java/jdk-18.0.1.1'
if not pt.started():
    pt.init()

SEED=42
RANDOM_STATE = 0

from textblob import TextBlob
import nltk
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import wordnet, stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize



# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import fastrank
# import lightgbm as lgb





stop_words = set(stopwords.words('english')) 
stemmer = PorterStemmer()
stop_words = stop_words.union({'http','https','www','href','src','img'})
stop_words = stop_words.union({'ll','re','ve'}) 
t_contractions = {'cant', 'wonnot', 'shouldnt', 'arent', 'daresnt', 'shallnt', 'wont', 'shant', 'darent', 'maynt', 'wat', \
    'oughtnt', 'couldnt', 'hadnt', 'didnt', 'yallrent', 'yaint', 'wasnt', 'dont', 'isnt', 'werent', 'doesnt', 'aint', 'havent', \
        'wouldnt', 'hasnt', 'neednt', 'mustnt', 'amnt', 'dasnt', 'mightnt', 'idnt', 'willnt'}
negations = {'no','not','cannot'}.union(t_contractions)

emotion_map = {'sadness':0, 'joy':1, 'love':2, 'anger':3,'fear':4,'surprise':5, 0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
emotions = ['sadness', 'joy', 'love', 'anger','fear','surprise']


def create_ngrams(text, nrange=(1, 1)):
    text_features = [] 
    text = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 1]
    text = [a[:-1] if a in t_contractions else a for a in text]
    
    for n in range(nrange[0], nrange[1]+1): 
        text_features += nltk.ngrams(text, n)
    return dict(Counter(text_features))   # Tf


def sentiment_textblob(row):
    classifier = TextBlob(row)
    polarity = classifier.sentiment.polarity
    subjectivity = classifier.sentiment.subjectivity
    return polarity,subjectivity



# credit for Loader class: https://stackoverflow.com/a/66558182
class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()






def main():
    print('\nEmotion Classifier and Search Engine')

    loader = Loader("Loading...", "Loading...done", 0).start()

    vectorizer = load('project_classify_models/vectorizer_0.25.joblib')
    clf = load( "project_classify_models/project_classify_LinearSVC.joblib")
    docs_df = pd.read_pickle('emotion/datasets/Emotion Dataset for Emotion Recognition Tasks/docs_df.pkl')

    # may require downloading nltk packages
    emotion_synonyms, emotion_antonyms = dict(), dict()
    for i in pd.unique(docs_df['emotion']):
        emotion_synonyms[i], emotion_antonyms[i] = set(), set()
        for syn in wordnet.synsets(i):
            for l in syn.lemmas():
                if '_' not in l.name():
                    emotion_synonyms[i].add(l.name())
                if l.antonyms() and '_' not in l.antonyms()[0].name():
                    emotion_antonyms[i].add(l.antonyms()[0].name())
    emotion_terms_all = {stemmer.stem(i) for a in emotion_synonyms for i in emotion_synonyms[a] }.union({stemmer.stem(i) for a in emotion_antonyms for i in emotion_antonyms[a] })


    pos_pref = ['NN', 'PR', 'VB', 'RB', 'MD', 'JJ', 'IN']
    pos_list = list(docs_df.pos.explode().unique())


    negations = {'no','not','arent', 'isnt', 'wasnt', 'werent', 'cant', 'couldnt', 'mustnt', 'shouldnt', 'wont', 'wouldnt', 'didnt', 'doesnt', 'dont', 'hasnt', 'havent', 'hadnt'}
    pos_pref = ['NN', 'PR', 'VB', 'RB', 'MD', 'JJ', 'IN']
    numeric_columns = ["polarity", "subjectivity", "er", "neg"] + pos_pref

    index_dir = os.path.join(os.getcwd(),'indices','carer7') #####
    index_ref = pt.IndexRef.of(os.path.join(index_dir, "data.properties"))
    index = pt.IndexFactory.of(index_ref)

    metadata = ['docno', 'emotion', 'polarity', 'subjectivity', 'er', 'neg', 'NN', 'PR', 'VB', 'RB', 'MD', 'JJ', 'IN']
    termpipelines = "PorterStemmer"
    # tf = pt.BatchRetrieve(index, metadata=metadata, wmodel="Tf", properties={"termpipelines" : termpipelines})
    # tfidf = pt.BatchRetrieve(index, metadata=metadata, wmodel="TF_IDF", properties={"termpipelines" : termpipelines})
    bm25 = pt.BatchRetrieve(index, metadata=metadata, wmodel="BM25", properties={"termpipelines" : termpipelines})

    ltr_feats1 = bm25 >> (
        pt.transformer.IdentityTransformer()
        ** 
        pt.FeaturesBatchRetrieve(index, wmodel="TF_IDF", features=["WMODEL:DirichletLM"], properties={"termpipelines" : ""})
        **
        pt.apply.doc_features(lambda row: np.array([float(row[a]) for a in numeric_columns]))
    )

    topics = docs_df[['emotion']].drop_duplicates(ignore_index=True).rename(columns={'emotion':'query'})
    topics['qid'] = [str(a) for a in range(topics.shape[0])]
    # topics

    qrels = docs_df.merge(topics, left_on='emotion', right_on='query', how='inner')[['qid','docno','label']]
    qrels.columns = ['qid','docno','label']
    # qrels

    num_queries_per_emotion = 100
    test_qrels = pd.DataFrame()
    for label in pd.unique(qrels.label):
        test_qrels = pd.concat([test_qrels, qrels.loc[qrels.label == label,:].sample(num_queries_per_emotion, random_state=0)], axis=0)

    tr_va_qrels = qrels.drop(test_qrels.index)
    train_qrels, valid_qrels =  train_test_split(tr_va_qrels, test_size=2/10, random_state=SEED)

    # importing only the best ML model
    # note that BM25 and TF-IDF both outperform it
    train_request = fastrank.TrainRequest.coordinate_ascent()
    params = train_request.params
    params.init_random = True
    params.normalize = True
    params.seed = 1234567
    ca_pipe = ltr_feats1 >> pt.ltr.apply_learned_model(train_request, form='fastrank')
    ca_pipe.fit(topics, train_qrels)

    loader.stop()


    def create_features_single_query(text, nrange=(1, 1)):
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        res = pd.DataFrame([text], columns=['text'])
        
        # copied from steps above, which were applied to docs_df
        res['tok'] = [word_tokenize(a) for a in res.text]
        res['stem'] = [[stemmer.stem(b) for b in a] for a in res.tok]
        res['pos_tup'] = [nltk.tag.pos_tag(a) for a in res.tok]
        res['pos'] = [[b[1] for b in a] for a in res.pos_tup]
        
        res[['polarity', 'subjectivity']] = pd.DataFrame([sentiment_textblob(a) for a in res.text], columns=['polarity', 'subjectivity'])

        for pref in pos_pref:
            tags = {a for a in pos_list if pref in a}
            res[pref] = [sum(1 for tag in row if tag in tags) for row in res.pos]
            
        res['pos_sum'] = res[pos_pref].sum(axis=1)
        res[pos_pref] = res[pos_pref].div(res['pos_sum'], axis=0)
        
        res['er'] = [sum(1 for tag in row if tag in emotion_terms_all) for row in res.stem] / res['pos_sum']
        res['neg'] = [sum(1 for tag in row if tag in negations) for row in res.tok] / res['pos_sum']
        
        
        ng = create_ngrams(res['text'].item(), nrange)
        res.columns = [tuple([a.upper()]) for a in res.columns]
        ret = [{**res[[tuple([a.upper()]) for a in numeric_columns]].to_dict(orient='index')[0], **ng}]
        return ret


    def predict_single_query(text, fitted_clf, vectorizer):
        feats = create_features_single_query(text, (1,4))
        print(feats)
        feats_vec = vectorizer.transform(feats)
        print(feats_vec)
        ret = fitted_clf.predict(feats_vec)
        return ret



    inp = ''
    while inp != 'exit':
        
        loader = Loader("", "", 0).start()
        sleep(0.5)
        loader.stop()
        
        
        print('\nWhat would you like to do? Choices:')
        print("   1 -> Classify emotion and display similar messages from the corpus")
        print("   2 -> Classify emotions of queries in a text file")
        print("   3 -> Get the emotion distribution about a topic from the corpus")
        print("exit -> exit program")
        
    
        inp = input("\nEnter choice -> ")
        if inp == '1':
            inp = input("Enter query -> ")
            inp = re.sub(r'[0-9]', '', inp).translate(str.maketrans('', '', string.punctuation))
            num_messages = int(input("Enter number of messages to retrieve (minimum is 20) -> "))
            num_messages = max(20, num_messages)
            predicted_emotion = predict_single_query(inp, clf, vectorizer)[0]
            messages = ca_pipe.search(inp)
            
            if messages.empty:
                # PyTerrier bug: https://github.com/terrier-org/pyterrier/issues/352
                print('Sorry! No results found. Try something else.')
                continue
            
            messages = pd.DataFrame(messages.head(num_messages)[['docno','emotion']].merge(docs_df[['docno','text']], on='docno'))
            
            print('Predicted emotion: ',predicted_emotion)
            print('Similar messages: ')
            print(''.rjust(10, ' ')+' Emotion'.ljust(12, ' ') + '\tText')
            
            # print(messages)
            for msg in messages.itertuples():
                print((str(msg.Index+1) + ' ').rjust(10, ' ')+' ' + str(msg.emotion).ljust(12, ' ') + '\t' + msg.text)
                
        
        elif inp == '2':
            
            
            print('Please ensure that each line in the file contains a single query.')
            inp = input("Enter relative path to file -> ")
            
            if not os.path.exists(inp):
                print('File not found! Try something else.')
                continue
            
            # test_input/test1.txt
                        
            with open(inp, 'r') as file:
                lines = file.readlines()
            
            lines = [re.sub(r'[0-9]', '', a).translate(str.maketrans('', '', string.punctuation)).replace('\n','') for a in lines]
            
        
            results = [predict_single_query(a, clf, vectorizer)[0] for a in lines]
                
            messages = pd.DataFrame(lines,columns=['text'])
            messages['emotion'] = results
            
            for msg in messages.itertuples():
                print((str(msg.Index+1) + ' ').rjust(10, ' ')+' ' + str(msg.emotion).ljust(12, ' ') + '\t' + msg.text)
            
            
            res = pd.DataFrame(100 * messages[['emotion','text']].groupby('emotion').count() / messages.shape[0]).reset_index()
            res.columns = ['emotion','percent']
            print(res)
            
            sns.set(rc={"figure.figsize":(5,5)})
            sns.histplot(messages, x='emotion' , stat="percent")
            print('See popup for graph')
            plt.show()
            # for line in lines:
            #     print(line)
                        
        
        
        elif inp == '3':
            inp = input("Enter query -> ")
            inp = re.sub(r'[0-9]', '', inp).translate(str.maketrans('', '', string.punctuation))
            num_messages = int(input("Enter number of messages to retrieve (minimum is 20) -> "))
            num_messages = max(20, num_messages)
            # predicted_emotion = predict_single_query(inp, clf, vectorizer)[0]
            messages = ca_pipe.search(inp)
            
            if messages.empty:
                # PyTerrier bug: https://github.com/terrier-org/pyterrier/issues/352
                print('Sorry! No results found. Try something else.')
                continue
            
            messages = pd.DataFrame(messages.head(num_messages)[['docno','emotion']].merge(docs_df[['docno','text']], on='docno'))
            
            res = pd.DataFrame(100 * messages[['emotion','text']].groupby('emotion').count() / messages.shape[0]).reset_index()
            res.columns = ['emotion','percent']
            print(res)
            
            sns.set(rc={"figure.figsize":(5,5)})
            sns.histplot(messages, x='emotion' , stat="percent")
            print('See popup for graph')
            plt.show()
            
        
        
            
            
                    
        elif inp.lower() == 'exit':
            print(inp)
            break
            
        else:
            print('Invalid input. Try again')
            continue
        



if __name__ == "__main__":
    main()





