import re,os,codecs,string
from bs4 import BeautifulSoup
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction import text 
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from wordcloud import WordCloud
from textblob import TextBlob
from gensim import matutils, models
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
class notes_library():
    
    color_word = re.compile(r'yellow|blue|pink|orange')

    stop_words = ['said','like','just','oh','say','know','says'] 
    stop_words_cap = []
    for i in text.ENGLISH_STOP_WORDS:
        stop_words.append(i)
    for i in stop_words:
        stop_words_cap.append(i.capitalize())
    contractions = ['ain','aren','couldn','didn','doesn','don','hadn','hasn','haven','I','isn','let','ma','mayn','mightn','mustn',
     'needn','o','oughtn','shan','sha','shouldn','wasn','weren','won','wouldn','y']
    stop_words = stop_words + contractions
    

    
    book_list=[]
    Kindle = os.listdir()
    for i in Kindle:
        if i.endswith('.html'):
            book_list.append(i)

    chapters=[]
    highlights=[]
    transcript=[]
    library = {}
    notes = []
    authors = []
    titles = []


    for name in book_list:
        f = codecs.open(name, 'r', 'utf-8')
        soup = BeautifulSoup(f, 'html.parser')
        title = soup.find('div', class_='bookTitle')
        author = soup.find('div', class_='authors')
        for i in soup.find_all('div',class_='sectionHeading'):
            chapters.append(i.text.strip())
        for i in soup.find_all('div',class_='noteHeading'):
            highlights.append(i.text.strip())
        for i in soup.find_all('div',class_='noteText'):
            transcript.append(i.text.strip())

        x = str(soup.get_text())
        lst = []

        title = title.text.strip()
        title = re.sub(r'\((.+)\)','',title)
        title = title.strip()
        author =  author.text.strip()#.split()[0][:-1]


        y =re.compile(r'(.*)\r')
        for i in y.findall(x):
            i = i.strip()
            if i != '':
                lst.append(i)
        for i in range(len(lst)):
            if lst[i] == author:
                notes = lst[i+1:]

        author = author.split()[0][:-1]
        if author not in authors:
            authors.append(author)
        titles.append(title)
        
        
        if author not in library.keys():
            library[author] = {}
        if title not in library[author].keys():
                library[author][title] = {}

        errors = []

        for i in range(len(notes)):
            if notes[i] in chapters:
                try:
                    g = i
                    library[author][title][notes[g]]={}
                    library[author][title][notes[g]][notes[i+1]]= [notes[i+2]]
                    k=g+3
                    if k in range(len(notes)) and notes[k] in highlights:
                        try:
                            while notes[k] in highlights: 
                                library[author][title][notes[g]][notes[k]]= [notes[k+1]]
                                k = k+2
                        except IndexError:
                            break
                except IndexError:
                    break
         
    colors = ['yellow','blue','orange','pink']
    

    
#### Clean Transcript ##########################################################################################    
    def clean_lst_to_text(self, lst):
        clean = self.lst_str(lst)
        clean = self.clean_text(clean)
        return clean
    
    
    def clean_text(self, texts):
        clean = texts
        clean = clean.lower()
        clean = re.sub(r'\[','',clean)
        clean = re.sub(r'[%s]' % re.escape(string.punctuation),'',clean)
        clean = re.sub(r'“|”','',clean)
        clean = re.sub(r'\w*\d\w*','',clean)
        clean = re.sub(r'\w*\'\w*','',clean)
        clean = re.sub(r'.\\xa0.\\xa0|\\xa0','',clean)
        return clean
    
    def clean_text_round1(self, text):
        text = re.sub(r'\[','',text)
        text = re.sub(r'\]','',text)
        text = re.sub(r',|','',text)
        text = re.sub(r'“|”','',text)
        text = re.sub(r'\w*\d\w*','',text)
        text = re.sub(r'.\\xa0.\\xa0|\\xa0|.\\.\\.','',text)
        text = re.sub(r'-',' ',text)
        return text

    def clean_word_round1(self, word):
        word = re.sub('[%s]' % re.escape(string.punctuation),'',word)
        return word

#### List to Text ##########################################################################################     
    def lst_str(self,lst):
        from itertools import chain
        lst = str(list(chain.from_iterable(lst)))
        return lst
    
#### Corpus Start ########################################################################################## 
    def corpus_start(self):
        x =[]
        for author in self.library:
            for title in self.library[author]:
                for chp in self.library[author][title]:
                    for loc in self.library[author][title][chp]:
                        col = self.color_word.search(loc)
                        if col is None:
                            break
                        for text in self.library[author][title][chp][loc]:
                            x.append([author,title,col.group(),text])
        h = pd.DataFrame(x, columns=(['Author','Title','Color','Text']))
        h = h.set_index(['Author','Title','Color'])
        return h


#### Corpus Spefic ##########################################################################################     
    def corpus_atc(self,author = False,title = False,color = False):
        lst = []
        parameters = [author,title,color]
        for para in range(len(parameters)):
            if type(parameters[para]) != list:
                parameters[para] = [parameters[para]]
        
        for a in parameters[0]:
            for t in parameters[1]:
                for c in parameters[2]:
                    x = self.corpus_single(a,t,c)
                    if len(x) == 0:
                        pass
                    else:
                        lst.append(x)
        return self.corpus_return_multi(lst)
        
    
#### Corpus Single ##########################################################################################  
    def corpus_single(self,author= False, title = False,color = False):
        corpusi = self.corpus_start()
        if author is False and title is False:
            texts = corpusi[(corpusi.index.get_level_values('Color').isin([color]))].Text
        elif title is False and color is False:
            texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))].Text
        elif author is False and color is False:
            texts = corpusi[(corpusi.index.get_level_values('Title').isin([title]))].Text
        elif author is False:
            texts = corpusi[(corpusi.index.get_level_values('Title').isin([title]))&(corpusi.index.get_level_values('Color').isin([color]))].Text
        elif title is False:
            texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))& (corpusi.index.get_level_values('Color').isin([color]))].Text
        elif color is False:
            texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))&(corpusi.index.get_level_values('Title').isin([title]))].Text
        else:
            texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))& (corpusi.index.get_level_values('Title').isin([title]))&(corpusi.index.get_level_values('Color').isin([color]))].Text

        return texts

#### Corpus return ##########################################################################################            
    def corpus_return(self,texts):
        lst=[]
        for i in texts:
            lst.append(i)
        return lst
    
#### Corpus return multiple series ######################################################################            
    def corpus_return_multi(self,texts):
        lst=[]
        for ii in texts:
            for i in ii:
                lst.append(i)
        return str(lst)
    
#### Stemming ######################################################################################
    def stem(self,lst):
        ps = PorterStemmer()
        words=[]
        for i in lst:
            words.append(ps.stem(i))
        return words


#### Document Term Matrix ######################################################################################
    
    def dtm(self,author=False,title=False,color=False):
        texts = self.clean_text(self.corpus_atc(author,title,color))
        docs = word_tokenize(texts)
        vec = CountVectorizer(stop_words = self.stop_words)
        X = vec.fit_transform(docs)
        array = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        dtm = pd.DataFrame(array.sum(),columns=['Text'])
        return dtm
    
#### WordCloud ######################################################################################            
    def wordcloud(self,author = False,title= False,color=False):
        lst = self.clean_text(self.corpus_atc(author,title,color))
        wc = WordCloud(stopwords = self.stop_words, background_color= 'white',colormap='Dark2',
                      max_font_size=150, random_state=42)
        w = wc.generate(lst)
        plt.imshow(w)
        plt.axis('off')
        plt.show()
        
#### Words info ######################################################################################   
    def words(self,author=False,title=False,color=False):
        info = {}
        info['number_of_words']= len(self.corpus_atc(author,title,color).split())
        info['unique_words'] = len(self.dtm(author,title,color).index)
        info['polarity'] = self.polarity(author,title,color)
        info['subjectivity']=self.subjectivity(author,title,color)
        return info
    
#### Polarity and Subjectivity ##############################################################################     
    def polarity(self,author=False,title=False,color=False):
        p = round(TextBlob(self.corpus_atc(author,title,color)).sentiment.polarity,4)
        return p
        
    def subjectivity(self,author,title,color):
        s = round(TextBlob(self.corpus_atc(author,title,color)).sentiment.subjectivity,4)
        return s

#### Bag of words ######################################################################################     
    def bow(self,author=False,title=False,color=False):
        lst=[]
        x = self.dtm(author,title,color).to_dict()
        for dic in x:
            for k,v in x[dic].items():
                if v != 1:
                    for v in range(v):
                        lst.append(k)
        else:
            lst.append(k)
        return lst

#### Topic Modeling ######################################################################################     
    def lda(self,author=False,title=False,color=False, number_of_topics = 5,passes =10):
        x = [self.bow(author,title,color)]
        id2word = corpora.Dictionary(x)
        corpus = [id2word.doc2bow(text) for text in x]        

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=number_of_topics, 
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=passes,
                                                   alpha='auto',
                                                   per_word_topics=True)

        return lda_model.print_topics()
    
#### Parts of speech ######################################################################################     
    def parts_of_speech(self,author=False,title=False,color=False,tag ='NN'):
        if type(tag) != list:
            tag = [tag]
        lst=[]
        tagged = pos_tag(self.bow(author,title,color))
        for word_t in range(len(tagged)):
            if tagged[word_t][1] in tag:
                lst.append(tagged[word_t][0])
        return lst
    
#### Pos LDA ######################################################################################         
    def lda_pos(self,author=False,title=False,color=False, number_of_topics = 3,passes =50,tag='NN'):
        x = [self.parts_of_speech(author,title,color)]
        id2word = corpora.Dictionary(x)
        corpus = [id2word.doc2bow(text) for text in x]        

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=number_of_topics, 
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=passes,
                                                   alpha='auto',
                                                   per_word_topics=True)

        return lda_model.print_topics()        


#### Character Loc ######################################################################################     
    def characters_locations(self,author=False,title=False,color=False,fantasy = 0):
        text = sent_tokenize(self.corpus_atc(author,title,color))
        char={}
        round1=[]
        for sent in text:
            sent = self.clean_text_round1(sent)
            words = sent.split()
            for w in range(len(words)):
                words[w] = self.clean_word_round1(words[w])
                if words[w].istitle() and w+1 in range(len(words)) and w !=0 and words[w] not in self.stop_words_cap:
                    if words[w] == 'Mr.' or words[w+1].istitle() and words[w+1] not in self.stop_words_cap:
                        words[w] = words[w] + words[w+1] 
                    round1.append(words[w])

        for word in round1:
            if round1.count(word) >= 1:
                char[word] = round1.count(word)

        char = pd.DataFrame(char.values(), columns = ['Number_of_Appearences'], index = char.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)
        if fantasy == 0:
            return char.head(round(char.shape[0]*.05))
        else:
            return char
    
    def fantasy_words(self,author=False,title=False,color=False):
        words = []
        look={}
        h=[]
        lst = self.dtm(author,title,color)
        spell = SpellChecker() 
        words = spell.unknown(lst.index.tolist())
        other = self.characters_locations(author,title,color,fantasy = 1).index.tolist()
        for i in words:
            if i.title() not in other and i not in self.contractions and lst.at[i,'Text'] !=1:
                    look[i] = lst.at[i,'Text']
        x = (look.keys())
        for i in x:
            if i.title() not in other and i not in self.contractions:
                h.append(i)
        
        y = self.stem(h)
        words_2 = spell.unknown(y)
        return words_2
        
        
NL = notes_library()
NL.dtm('Gaiman')