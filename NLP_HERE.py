import re,os,codecs,string
import math
import spacy
from itertools import chain
from bs4 import BeautifulSoup
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction import text 
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk import RegexpParser, ne_chunk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from wordcloud import WordCloud
from textblob import TextBlob
from gensim import matutils, models
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from ebooklib import epub
import ebooklib
import textract
import unicodedata

class Library():
    color_word = re.compile(r'yellow|blue|pink|orange')
    location_word = re.compile(r'Location')
    page_word = re.compile(r'Page')

    stop_words = ['said','like','just','oh','say','know','says','man','yes','uh'] 
    stop_words_cap = []
    for i in text.ENGLISH_STOP_WORDS:
        stop_words.append(i)
    for i in stop_words:
        stop_words_cap.append(i.capitalize())
    contractions = ['ain','aren','couldn','didn','doesn','don','hadn','hasn','haven','I','isn',
                    'let','ma','mayn','mightn','mustn','needn','o','oughtn','shan','sha','shouldn',
                    'wasn','weren','won','wouldn','y','ve','ll','im','ive','ill','thats','youre','dont','id']
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
        title = re.sub('\((.+)\)','',title)
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
    
    complete_library = {}
    complete_authors = []
    complete_titles =[]
    full_text_path = os.getcwd() +'/Full_Text_txt'
    for entry in os.scandir(full_text_path):
        if entry.name.endswith('.txt'):
            shelf = entry.name.split(' by ')
            complete_fullname_author = shelf[1][:-4]
            complete_author = complete_fullname_author.split(' ')[1]
            complete_authors.append(complete_author)
            complete_title = shelf[0]
            complete_titles.append(complete_title)
            with open(entry.path) as file:
                book = file.read()        
            if complete_author not in complete_library.keys():
                complete_library[complete_author] = {complete_title:book}
            else:
                complete_library[complete_author][complete_title] = book
    complete_authors = set(complete_authors)
    complete_titles = set(complete_titles)
#### Clean Transcript ######################################################################################    
    def clean_word_round2(self,word):
        word = re.sub(r'[%s]' % re.escape(string.punctuation),'',word)
        return word
    
    def clean_word_comma(self,word):
        word = re.sub(r'‘','',word)
        word = re.sub(r'’','',word)
        word = re.sub(r'“|”','',word)
        return word

    def no_questions(self,word):
        word = word.replace('?','.')
        word = word.replace('!','.')
        word = re.sub(r'‘','',word)
        word = re.sub(r'’',' ',word)
        word = re.sub(r'\'',' ',word)
        word = re.sub(r'“|”','',word)
        word = re.sub(r',',' ',word)
        word = re.sub(r'"','',word)
        word = re.sub(r'\\','',word)
        word = re.sub(r'…',' ',word)
        word = re.sub(r'—',' ',word)
        #word = re.sub(r'(|)',' ',word)
        return word
    
    
    def clean_numbers(self,text):
        text = [int(s) for s in re.findall(r'\b\d+\b', text)]
        for i in text:
            return i
    
    def clean_lst_to_text(self, lst):
        clean = self.lst_str(lst)
        clean = self.clean_text(clean)
        return clean
    
    def clean_word_cap(self, texts):
        clean = texts
        clean = re.sub(r'\[','',clean)
        clean = re.sub(r'[%s]' % re.escape(string.punctuation),'',clean)
        clean = re.sub(r'“|”','',clean)
        clean = re.sub(r'\w*\d\w*','',clean)
        clean = re.sub(r'\w*\'\w*','',clean)
        clean = re.sub(r'.\\xa0.\\xa0|\\xa0','',clean)
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
        #text = re.sub(r'“|”','',text)
        text = re.sub(r'\w*\d\w*','',text)
        text = re.sub(r'.\\xa0.\\xa0|\\xa0|.\\.\\.','',text)
        text = re.sub(r'-',' ',text)
        return text
    
    def remove_quotes(self,text):
        text = re.sub(r'“|”','',text)
        test = re.sub(r',','',text)
        return text
    
    def clean_word_round1(self, word):
        word = re.sub(r'[%s]' % re.escape(string.punctuation),'',word)
        word = re.sub(r'“|”','',word)
        return word
    
    def clean_word_round2(self, word):
        word = re.sub(r'[%s]' % re.escape(string.punctuation),'',word)
        return word
#### Unchain a List ##########################################################################################     
    def unchain(self,lst):
        lst = list(chain.from_iterable(lst))
        return lst  
#### List to Text ##########################################################################################     
    def lst_str(self,lst):
        lst = str(list(chain.from_iterable(lst)))
        return lst
#### library ##########################################################################################     

    def library_pick(self,lib = 'Notes'):
        if lib == 'Notes':
            lib = self.library
        else:
            lib = self.complete_library
        return lib  
#### Corpus Start Notes ##################################################################################### 
    def corpus_start_notes(self):
        x =[]
        for author in self.library:
            for title in self.library[author]:
                for chp in self.library[author][title]:
                    for loc in self.library[author][title][chp]:
                        col = self.color_word.search(loc)
                        if col is None:
                            break
                        page = self.page_word.search(loc)
                        if page is not None:
                            page_num = self.clean_numbers(loc[page.span()[1]+1:page.span()[1]+4])
                        else:
                            break
                        loca = self.location_word.search(loc)
                        if loca is not None:
                            location_num = self.clean_numbers(loc[loca.span()[1]+1:loca.span()[1]+6])
                        else:
                            break
                        for text in self.library[author][title][chp][loc]:
                            polarity = round(TextBlob(text).sentiment.polarity,4)
                            length = len(text)
                            x.append([author,title,col.group(),page_num,location_num,polarity,length,text])
                            
        h = pd.DataFrame(x, columns=(['Author','Title','Color','Page','Location','Polarity','Length','Text']))
        h = h.set_index(['Author','Title','Color','Page'])
        return h
#### Corpus Start Full ########################################################################################
    def corpus_start_full(self):
        x =[]
        for author in self.complete_library:
            for title in self.complete_library[author]:
                text = self.complete_library[author][title]
                polarity = round(TextBlob(text).sentiment.polarity,4)
                length = len(text)
                x.append([author,title,polarity,length,text])

        h = pd.DataFrame(x, columns=(['Author','Title','Polarity','Length','Text']))
        h = h.set_index(['Author','Title'])
        return h
#### Corpus Spefic ##########################################################################################     
    def corpus_atc(self,author = False,title = False,color = False,library ='Notes',lem=False):
        lst = []
        parameters = [author,title,color]
        for para in range(len(parameters)):
            if type(parameters[para]) != list:
                parameters[para] = [parameters[para]]
        
        for a in parameters[0]:
            for t in parameters[1]:
                for c in parameters[2]:
                    x = self.corpus_single(a,t,c,library)
                    if len(x) == 0:
                        pass
                    else:
                        lst.append(x)
        if lem:
            corpus = ''
            #words = word_tokenize(self.corpus_return_multi(lst))
            words = self.corpus_return_multi(lst).split(' ')
            for i in range(len(words)):
                words[i] = self.lemmatizer(words[i])
                if i == 0:
                    corpus +=words[i]
                else:
                    corpus = corpus + ' '+ words[i]
            return corpus 
        else:
            return self.corpus_return_multi(lst)    
#### Corpus Spefic Pretty ##################################################################################     
    def corpus_atc_pretty(self,author = False,title = False,color = False,library='Notes'):
        lst = []
        parameters = [author,title,color]
        for para in range(len(parameters)):
            if type(parameters[para]) != list:
                parameters[para] = [parameters[para]]
        
        for a in parameters[0]:
            for t in parameters[1]:
                for c in parameters[2]:
                    x = self.corpus_single(a,t,c,library)
                    if len(x) == 0:
                        pass
                    else:
                        lst.append(x)
        lst1=[]
        for ii in lst:
            for i in ii:
                lst1.append(i)
        for i in lst1:
            print(i+"\n")   
#### Corpus Single ##########################################################################################  
    def corpus_single(self,author= False, title = False,color = False,library ='Notes',get = 'Text' ):
        if library == 'Notes':
            corpusi = self.corpus_start_notes()
            if author == False and title == False and color == False:
                texts = corpusi[get]
            elif author is False and title is False:
                texts = corpusi[(corpusi.index.get_level_values('Color').isin([color]))][get]
            elif title is False and color is False:
                texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))][get]
            elif author is False and color is False:
                texts = corpusi[(corpusi.index.get_level_values('Title').isin([title]))][get]
            elif author is False:
                texts = corpusi[(corpusi.index.get_level_values('Title').isin([title]))&\
                      (corpusi.index.get_level_values('Color').isin([color]))][get]
            elif title is False:
                texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))&\
                      (corpusi.index.get_level_values('Color').isin([color]))][get]
            elif color is False:
                texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))&\
                      (corpusi.index.get_level_values('Title').isin([title]))][get]
            else:
                texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))&\
                          (corpusi.index.get_level_values('Title').isin([title]))&\
                          (corpusi.index.get_level_values('Color').isin([color]))][get]

            return texts
        else:
            corpusi = self.corpus_start_full()
            if author == False and title == False:
                texts =corpusi[get]
            elif author == False:
                texts = corpusi[(corpusi.index.get_level_values('Title').isin([title]))][get]
            elif title == False:
                texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))][get]
            else:
                texts = corpusi[(corpusi.index.get_level_values('Author').isin([author]))&\
                          (corpusi.index.get_level_values('Title').isin([title]))][get]
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
#### Document Term Matrix #################################################################################
    
    def dtm(self,author=False,title=False,color=False,library ='Notes',lem=False):
        texts = self.clean_text(self.corpus_atc(author,title,color,library,lem))
        docs = word_tokenize(texts)
        vec = CountVectorizer(stop_words = self.stop_words)
        X = vec.fit_transform(docs)
        array = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        dtm = pd.DataFrame(array.sum(),columns=['Text'])
        return dtm
#### WordCloud ######################################################################################            
    def wordcloud(self,author = False,title= False,color=False,library ='Notes',lem=False):
        lst = self.clean_text(self.corpus_atc(author,title,color,library,lem))
        wc = WordCloud(stopwords = self.stop_words, background_color= 'white',colormap='Dark2',
                      max_font_size=150, random_state=42)
        w = wc.generate(lst)
        plt.imshow(w)
        plt.axis('off')
        plt.show()       
#### Words info ######################################################################################   
    def words(self,author=False,title=False,color=False,library ='Notes',lem=False):
        info = {}
        info['number_of_words']= len(self.corpus_atc(author,title,color,library,lem).split())
        info['unique_words'] = len(self.dtm(author,title,color,library,lem).index)
        info['polarity'] = self.polarity(author,title,color,library)
        info['subjectivity']=self.subjectivity(author,title,color,library)
        return info
#### Polarity and Subjectivity ##############################################################################     
    def polarity(self,author=False,title=False,color=False,library ='Notes',lem=False):
        p = round(TextBlob(self.corpus_atc(author,title,color,library,lem)).sentiment.polarity,4)
        return p
        
    def subjectivity(self,author=False,title=False,color=False,library ='Notes',lem=False):
        s = round(TextBlob(self.corpus_atc(author,title,color,library,lem)).sentiment.subjectivity,4)
        return s
#### Lemmatizer ##############################################################################     
    def lemmatizer(self,word,lst=False):
        lemmatizer = WordNetLemmatizer()
        if lst:
            for i in range(len(word)):
                word[i]=lemmatizer.lemmatize(word[i])
            return word

        else:
            return lemmatizer.lemmatize(word)   
#### Bag of words ######################################################################################     
    def bow(self,author=False,title=False,color=False,library ='Notes',lem=False):
        lst=[]
        x = self.dtm(author,title,color,library,lem).to_dict()
        for dic in x:
            for k,v in x[dic].items():
                if v != 1:
                    for i in range(v):
                        lst.append(k)
        else:
            lst.append(k)
        return lst   
#### Parts of speech ######################################################################################     
    def parts_of_speech(self,author=False,title=False,color=False,library ='Notes',tag =['NN','NNS']):
        if type(tag) != list:
            tag = [tag]
        lst=[]
        for w in tag:
            if w == 'nouns':
                tag.remove('nouns')
                tag = tag +['NN','NNS','NNP','NNPS']
            elif w == 'verbs':
                tag.remove('verbs')
                tag + ['VB','VBD','VBG','VBN','VBP','VBZ']
            elif w == 'adjs':
                tag.remove('adjs')
                tag + ['JJ','JJR','JJS']
        tagged = pos_tag(self.bow(author,title,color,library))
        for word_t in range(len(tagged)):
            if tagged[word_t][1] in tag:
                lst.append(tagged[word_t][0])
        return lst  
#### LDA ######################################################################################         
    def lda(self,author=False,title=False,color=False,library ='Notes', topics = 3,passes =50,tag=False):
        if tag != False:
            x = [self.parts_of_speech(author,title,color,library,tag)]
        else:
            x = [self.bow(author,title,color,library)]
        id2word = corpora.Dictionary(x)
        corpus = [id2word.doc2bow(text) for text in x]        

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=topics, 
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=passes,
                                                   alpha='auto',
                                                   per_word_topics=True)

        return lda_model.print_topics()    
#### Lsi ######################################################################################         
    def lsi(self,author=False,title=False,color=False,library ='Notes', topics = 3,tag=False):
        if tag != False:
            x = [self.parts_of_speech(author,title,color,library,tag)]
        else:
            x = [self.bow(author,title,color,library)]
        id2word = corpora.Dictionary(x)
        corpus = [id2word.doc2bow(text) for text in x]        

        lsi_model = gensim.models.lsimodel.LsiModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=topics, 
                                                   chunksize=100)

        return lsi_model.print_topics()
#### Pie Chart ######################################################################################         
    def pie_chart(self,author=False,title=False,color=False,library ='Notes'):
        corpusi = self.corpus_single(author,title,color,library,get='Length')
        pie_title = str(author) +': '+ str(title)
        colors = sorted(list(set(corpusi.index.get_level_values('Color').tolist())))
        return corpusi.groupby(['Color']).sum().plot(kind='pie',
                                                     y='Length',
                                                     colors = colors,
                                                     title=pie_title.replace('False',''),
                                                     legend = False,
                                                     autopct='%1.1f%%',
                                                     figsize = (7,7))
    
    def fantasy_words(self,author=False,title=False,color=False,library ='Notes',lem=False):
        words = []
        look={}
        h=[]
        lst = self.dtm(author,title,color,library,lem)
        spell = SpellChecker() 
        words = spell.unknown(lst.index.tolist())
        other = self.characters_locations(author,title,color,library,lem).index.tolist()
        for i in words:
            if i.title() not in other and i not in self.contractions and lst.at[i,'Text'] !=1:
                    look[i] = lst.at[i,'Text']
        x = (look.keys())
        for i in x:
            if i.title() not in other and i not in self.contractions:
                h.append(i)
        
        #y = self.stem(h)
        #words_2 = spell.unknown(y)
        return h
    
    def count_characters_repeats(self):
        from collections import Counter
        repeats =[]
        for author in self.authors:
            for character in self.characters_locations(author = author, fantasy = True).index:
                repeats.append(character)
        repeats = Counter(repeats)
        repeats = pd.DataFrame(repeats.values(), columns = ['Occurences'],\
                               index = repeats.keys()).sort_values(by='Occurences',ascending=False)
        repeats = repeats[repeats['Occurences'] >= 2]
        return repeats

    def parts_of_speech0(self,author=False,title=False,color=False,library ='Notes',lem=False,tag ='NN'):
        if type(tag) != list:
            tag = [tag]
        lst=[]
        x = self.corpus_atc(author,title,color,library,lem).split()
        for w in x:
            self.clean_word_round1(w)
        tagged = pos_tag(x)
        for word_t in range(len(tagged)):
            if tagged[word_t][1] in tag:
                lst.append(tagged[word_t][0])
        return lst  
        
    def char(self,author=False,title=False,color=False,library ='Notes',lem=False,fantasy = 0):
        lst=[]
        char={}
        text = word_tokenize(self.corpus_atc(author,title,color,library,lem))
        tagged = pos_tag(text)
        for word_t in range(len(tagged)):
            if tagged[word_t][1] == 'NNP':
                lst.append(tagged[word_t][0])
        for word in lst:
            word = self.clean_text(word).title()
        tagged_two = pos_tag(lst)
        for word in tagged_two:
            char[word[0]] = tagged_two.count(word)
        return char
    
    def both_char(self,author=False,title=False,color=False,library ='Notes'):
        lst=[]
        other=[]
        char_1 = self.characters_locations(author,title,color,library,fantasy=True).index
        for character in self.char(author,title,color,library):
            if character in char_1:
                lst.append(character)
            else:
                other.append(character)
                
        return lst        
    
    def describe_char(self,author=False,title=False,color=False,library ='Notes',lem=False):
        x={}
        chars = self.characters_locations(author,title,color,library,fantasy=0).index.tolist()
        for characters in chars:
            x[characters] = []
        words = word_tokenize(self.corpus_atc(author,title,color,library,lem))
        for w in range(len(words)):
            words[w] = self.clean_word_cap(words[w])
        words = list(filter(('').__ne__, words))
        tagged = pos_tag(words)
        for character in chars:
            for words in range(len(tagged)):
                if character == tagged[words][0] and tagged[words-1][1] == 'JJ':
                    x[character].append(tagged[words-1][0])
                    
        
        return x
    
    def char_actions(self,author=False,title=False,color=False,library ='Notes',lem=False):
        x={}
        chars = self.characters_locations(author,title,color,library,fantasy=0).index.tolist()
        for characters in chars:
            x[characters] = []
        words = word_tokenize(self.corpus_atc(author,title,color,library,lem))
        for w in range(len(words)):
            words[w] = self.clean_word_cap(words[w])
        words = list(filter(('').__ne__, words))
        tagged = pos_tag(words)
        for character in chars:
            for words in range(len(tagged)):
                if character == tagged[words][0] and tagged[words-1][1].startswith('V') and\
                tagged[words-1][0] not in self.stop_words:
                    x[character].append(tagged[words-1][0])
                elif character == tagged[words][0] and tagged[words+1][1].startswith('V') and\
                tagged[words+1][0] not in self.stop_words:
                    x[character].append(tagged[words-1][0])
                elif character == tagged[words][0] and tagged[words+1][1] == 'DT' and\
                tagged[words+2][1].startswith('V') and tagged[words+1][0] not in self.stop_words:
                    x[character].append(tagged[words-1][0])
        return x
    
    def testing(self,author=False,title=False,color=False,library ='Notes',lem=False):
        lst=[]
        pattern = 'NP: {<DT>?<JJ>*<NN>}'
        cp = RegexpParser(pattern)
        sent = pos_tag(word_tokenize(self.corpus_atc(author,title,color,library,lem)))
        cp = RegexpParser(pattern)
        cs = cp.parse(sent)
        print(cs)
        #for s in sent:
            #lst.append(cp.parse(s))
            
    def count_pos(self,sr,pos):
        x = {}
        for index, text in sr.items():
            x[index] = 0
            for word in pos_tag(word_tokenize(text)):
                if word[1].startswith(pos):
                    x[index] +=1
        return pd.Series(x)
    
    def dialoge(self,sr):
        x={}
        quotes = re.compile(r'“[^”]+”')
        #quotes = re.compile(r'“\s*((?:\w(?!\s+”)+|\s(?!\s*“))+\w)\s*”')
        for index,text in sr.items():
            dialoge = quotes.findall(text)
            x[index] = str(dialoge)
        return pd.Series(x)
    
    
    def characters__(self,sr):
        x={}
        for index,text in sr.items():
            y=self.characters_locations(text=text).index.values
            x[index] = str(y)[1:-1]
            
        return pd.Series(x)
            
            
    def corpus_expand(self):
        x=self.corpus_start_notes()
        pos_terms={'N':'noun_count','V':'verb_count','J':'adjective_counts'}
        for k,v in pos_terms.items():
            x[v] = self.count_pos(x['Text'],k)
        x['Diagloge'] = self.dialoge(x['Text'])
        x['Characters_Locations'] = self.characters__(x['Text'])
        #x['Text_Length'] = x['Text'].str.len()
        return x
    
    def character_search_notes(self,author = False,title=False,color=False,library='Notes'\
                               ,find='Shadow',get =False):
        #Improve to add Full and list of finds
        corpus = self.corpus_expand()
        corpus = corpus[corpus.Characters_Locations.str.contains(find)]
        if get == False:
            return corpus
        else:
            return corpus[get]
        
    def character_color_wordcloud(self,author = False,title=False,color=False,library='Notes',find='Shadow'):
        corpusi = self.character_search_notes(author,title,color,library,find,get='Length')
        pie_title = find
        colors = sorted(list(set(corpusi.index.get_level_values('Color').tolist())))
        return corpusi.groupby(['Color']).sum().plot(kind='pie',
                                                     y='Length',
                                                     colors = colors,
                                                     title=pie_title.replace('False',''),
                                                     legend = False,
                                                     autopct='%1.1f%%',
                                                     figsize = (7,7))   
#### Character Loc ######################################################################################  
#### Cutoff only works with Full Library
    def characters_locations(self,author=False,title=False,color=False,library ='Notes',lem=False,\
                             text = False,expand=False):
        if text == False:
            corpus = self.no_questions(self.corpus_atc(author,title,color,library,lem))
            sentances = sent_tokenize(corpus)
        else:
            sentances = sent_tokenize(text)
        connectors = ['au','dan']
        ofs = ['of','of the']
        and_dic={}
        of_dic={}
        connector_dic = {}
        new_characters={}
        
        y=0
        first_word = {}
        space ={}
        for sent in sentances:
            sent = sent.strip()
            words = word_tokenize(sent)
            for w in range(len(words)):
                if words[w].istitle() and words[w] not in self.stop_words_cap and w !=0:
                    if words[w] in new_characters.keys():
                        new_characters[words[w]] +=1
                    else:
                        new_characters[words[w]] = 1 
                    while w+1 in range(len(words)) and words[w+1].istitle() and\
                    words[w+1] not in L.stop_words_cap and words[w+1] not in L.authors:
                        words[w] = words[w] + ' ' +words[w+1]
                        if w+2 in range(len(words)) and words[w+2].istitle() and\
                        words[w+2] not in L.stop_words_cap and words[w+1] not in L.authors:
                            words[w+1]=words[w]
                            w += 2
                        else:
                            if words[w] in space.keys():
                                space[words[w]] +=1
                            else:
                                space[words[w]] = 1
                            words[w+1] = ''
                            break
                    if w+2 in range(len(words)) and words[w+1] in connectors and words[w+2].istitle():
                        words[w] = words[w] + ' ' + words[w+1] + ' ' + words[w+2]
                        if words[w] in connector_dic.keys():
                                connector_dic[words[w]] +=1
                        else:
                            connector_dic[words[w]] = 1
                    if w+2 in range(len(words)) and 'and' in words[w+1] and words[w+2].istitle():
                        words[w] = words[w] + ' ' + words[w+1] + ' ' + words[w+2]
                        if words[w] in and_dic.keys():
                                and_dic[words[w]] +=1
                        else:
                            and_dic[words[w]] = 1
                    if w+2 in range(len(words)) and words[w+1] in ofs and words[w+2].istitle():
                        words[w] = words[w] + ' ' + words[w+1] + ' ' + words[w+2]
                        if words[w] in of_dic.keys():
                                of_dic[words[w]] +=1
                        else:
                            of_dic[words[w]] = 1
                elif w == 0:
                    if words[w] in first_word.keys():
                        first_word[words[w]] +=1
                    else:
                        first_word[words[w]] = 1 
                    
                        
        char = pd.DataFrame(new_characters.values(), columns = ['Number_of_Appearences'], 
                        index = new_characters.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)
        first_word = pd.DataFrame(first_word.values(), columns = ['Number_of_Appearences'],
                          index = first_word.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)

        z = first_word.join(char,lsuffix='_first', rsuffix='_char',how='inner')
        z['Total'] = z['Number_of_Appearences_first'] + z['Number_of_Appearences_char']
        #z['Difference'] = z['Number_of_Appearences_first'] - z['Number_of_Appearences_char']
        z['Percent'] = z['Number_of_Appearences_first'] / z['Number_of_Appearences_char']
        z.index.name = 'Characters'
        percent_cutoff = z.Percent.mean()
        if library != 'Notes':
            z_=z[z['Percent']>percent_cutoff]
            z= z[z['Percent']<percent_cutoff]
            
        space = pd.DataFrame(space.values(), columns = ['Number_of_Appearences'], 
                    index = space.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)
        space.index.name ='Space Characters'
        space = space[space['Number_of_Appearences'] != 1]
        
        connector_dic = pd.DataFrame(connector_dic.values(), columns = ['Number_of_Appearences'], 
                    index = connector_dic.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)

        and_dic = pd.DataFrame(and_dic.values(), columns = ['Number_of_Appearences'], 
                            index = and_dic.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)
        #and_dic = and_dic[and_dic['Number_of_Appearences'] != 1]

        of_dic = pd.DataFrame(of_dic.values(), columns = ['Number_of_Appearences'], 
                            index = of_dic.keys()).sort_values(by= 'Number_of_Appearences',ascending =False)
        of_dic = of_dic[of_dic['Number_of_Appearences'] != 1]

        space = pd.concat([space,of_dic])
        #nine_eight = z[z['Total'] > z.quantile(.98)['Total']]
        
        if expand:
            return [z,space,connector_dic,and_dic,of_dic]
        else:
            return z
#### Character Loc Expand ##################################################################################  
#### Cutoff only works with Full Library    
    def characters_locations_expand(self,author=False,title=False,color=False,\
                                    library ='FL',lem=False,expand = False):
        expand_lst = self.characters_locations(author,title,color,library,lem,expand=True)
        space = expand_lst[1]
        contains_space = space.index.tolist()
        one_percent = expand_lst[0]#.head(math.ceil(z.shape[0]*.25))
        skip_lst = expand_lst[3].index
        x={}
        y=[]
        pop_lst=[]
        skip_lst=[]

        for name in one_percent.index.tolist():
            x[name] = {'Prefix':{},'Suffix':{}}
            for s_name in contains_space:
                appearences = int(space.loc[s_name].Number_of_Appearences.item())
                if name in s_name and appearences != 1:
                    if s_name.startswith(name+' ') and s_name not in skip_lst :
                        x[name]['Prefix'][s_name] = appearences
                    if s_name.endswith(' '+name) and s_name not in skip_lst :
                        x[name]['Suffix'][s_name] = appearences

        for name in x.keys():
            p_most = [0,'',0]
            s_most = [0,'',0]
            blue = {'Name':name,'All_Names':[]}
            blue['Prefix_Count'] = len(x[name]['Prefix'])
            for p_name in x[name]['Prefix'].keys():
                if p_most[0] == 0:
                    p_most[0] = x[name]['Prefix'][p_name]
                    p_most[1] = p_name
                elif x[name]['Prefix'][p_name] > p_most[0]:
                    p_most[0] = x[name]['Prefix'][p_name]
                    p_most[1] = p_name
                p_most[2] += x[name]['Prefix'][p_name]
                blue['All_Names'].append(p_name)
            blue['Prefix_Appearences'] = p_most[2]
            blue['Prefix_Max'] = p_most[0]
            blue['Prefix_Max_Name'] = p_most[1]
            blue['Suffix_Count'] = len(x[name]['Suffix'])
            for s_name in x[name]['Suffix'].keys():
                if s_most[0] == 0:
                    s_most[0] = x[name]['Suffix'][s_name]
                    s_most[1] = s_name
                elif x[name]['Suffix'][s_name] > s_most[0]:
                    s_most[0] = x[name]['Suffix'][s_name]
                    s_most[1] = s_name
                s_most[2] += x[name]['Suffix'][s_name] 
                blue['All_Names'].append(s_name)
            blue['Suffix_Appearences'] = s_most[2]
            blue['Suffix_Max'] = s_most[0]
            blue['Suffix_Max_Name'] = s_most[1]
            blue['Total_Appearences'] = blue['Suffix_Appearences'] +blue['Prefix_Appearences']
            blue['Total_Count'] = blue['Suffix_Count'] + blue['Prefix_Count']
            if blue['Suffix_Max'] > blue['Prefix_Max']:
                blue['Total_Max_Name'] = blue['Suffix_Max_Name']
            elif blue['Suffix_Max'] == blue['Prefix_Max']:
                blue['Total_Max_Name'] = blue['Prefix_Max_Name']+' - '+blue['Suffix_Max_Name']
            else:
                blue['Total_Max_Name'] = blue['Prefix_Max_Name']

            y.append(blue)




        y = pd.DataFrame(y).sort_values('Total_Appearences',ascending=False)
        y.set_index('Name', inplace =True)
        return y


L = Library()

