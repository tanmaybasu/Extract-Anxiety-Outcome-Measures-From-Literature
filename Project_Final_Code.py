# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:42:31 2018

@author: User
"""
import time
start_time = time.time()
from sklearn.datasets import load_files
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import string
import fitz
import os
import re
import nltk

from sklearn import svm
from sklearn import grid_search  
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_selection import SelectKBest,chi2
from os.path import isfile, join
from nltk.tag import StanfordNERTagger




#################converting pdf files into txt and cleaning text files ##########################################
'''
This function converts the pdf files into text files and cleans them while converting.
The cleaning is done using different regular expressions. The regular expressions use different
expressions as required to clean the data. The expressions used are as per the current data.
If in future more files come, then further cleaning will need to be handled then.
'''
def pdftotxt(base_path,path1):
        lists=os.listdir(base_path)
        count_tot=len(lists)
        print ("The total number of files in",base_path," are: ",count_tot)
        for filename in os.listdir(base_path):
            texts=[]
            text=""
            pathpdf=os.path.join(base_path+"/"+filename)
            if filename.endswith(".pdf" or ".xps" or ".txt") :
                    doc = fitz.open(pathpdf)
                    pages=doc.pageCount
                    for i in range(pages):
                        page = doc.loadPage(i)
                        texts.append(page.getText(output='text'))
                    text=" ".join(texts)
                    
            if filename.endswith(".txt"):
                f1=open(pathpdf,encoding="utf-8")
                doc=f1.read()
                text=doc         
                        
                    
            m=0
            pat1 = r'\.'
            for ar in re.finditer(pat1,text):
                d=ar.start()+m
                if(d<(len(text)-1)):
                    if(text[d+1].isspace() is False and text[d+1].isnumeric() is False):
                        temp = text[:d+1]+' '+text[d+1:]
                        text=temp
                        m=m+1
                        
            m=0            
                        
            pat = r'\.'
            for ar in re.finditer(pat1,text):
                d=ar.start()+m
                if(d<(len(text)-1)):
                    if(text[d+1].isnumeric() is True and text[d-1].isalpha() is True):
                        temp = text[:d+1]+' '+text[d+1:]
                        text=temp
                        m=m+1       
                        
            pat2 = '�'
            for ar in re.finditer(pat2,text):
                d=ar.start()
                if(d<(len(text)-1)):
                    if(text[d+1].isnumeric() and text[d-1].isnumeric):
                        temp = text[:d]+'.'+text[d+1:]
                        text=temp
                                  
            text=re.sub(r'‡','>=',text)   
            text=re.sub(r'Æ','.',text)
            text=re.sub(r'\[','>',text)
            text=re.sub(r'¼','=',text)
            
            pat4=r'e'
            for ar in re.finditer(pat4,text):
                d=ar.start()
                if(d<(len(text)-1)):
                    if(text[d+1].isnumeric() and text[d-1].isnumeric):
                        temp = text[:d]+'-'+text[d+1:]
                        text=temp
                       
            pat = r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+'
            text = re.sub(pat,' ', text)
            
            pat3 = r'\n'
            for ar in re.finditer(pat3,text):
                d=ar.start()
                if(d<(len(text)-1)):
                    if(text[d+1].islower() or text[d+1].isnumeric() or (text[d+1] in string.punctuation)):
                        temp = text[:d]+' '+text[d+1:]
                        text=temp
                        
            patn=re.compile("i\.e\.|eg\.|ie\.|e\.g\.|et al\.|�")            
            text=re.sub(patn,"",text)            
            
            pat_tab=re.compile('table \d+',re.I)
            for ar in re.finditer(pat_tab,text):
                d=ar.end()
                print (text[d],filename)
                if(d<(len(text)-1)):
                    if(text[d]=='.' and (text[d+1].isspace() or text[d+1].isalpha())):
                        temp = text[:d]+' '+text[d+1:]
                        text=temp
            
            
            if (text):
                txtfilename=os.path.splitext(filename)[0]
                log_file = os.path.join(path1 + "/" + txtfilename + ".txt")
                with open(log_file, "wb") as my_log:
                    my_log.write(text.encode("utf-8"))
                print("Done !!")
            else:
                print ("File could not be read or could not be converted !!")


#####--------------FUnction for taking user choice and path and calling pdftotext()----------#####
'''
This is a user choice function. Depending on the user input, the convert the pdf files from 
the respective folders.
'''
def pdftotxtchoice():    
    print("Do you want to convert training anxiety files into text? Enter Y for yes else N for no")
    choice=input()
    if (choice=="Y" or choice=="y"):
        base_path= r"D:\Msc Gogol books\ML\Summer Project\Anxiety\Train PDF"
        path1=r'D:\Msc Gogol books\ML\Summer Project\Anxiety\Train PDF\Text'
        pdftotxt(base_path,path1)
        
    print("Do you want to convert training non anxiety files into text? Enter Y for yes else N for no")
    choice=input()
    if (choice=="Y" or choice=="y"):    
         base_path_na= r"D:\Msc Gogol books\ML\Summer Project\Anxiety\Train PDF\Non-Anxiety"
         path2=r'D:\Msc Gogol books\ML\Summer Project\Anxiety\Train PDF\Text\Non Anxiety'
         pdftotxt(base_path_na,path2)
                  
    print("Do you want to convert test files into text? Enter Y for yes else N for no")
    choice=input()
    if (choice=="Y" or choice=="y"):   
        base_path= r"D:\Msc Gogol books\ML\Summer Project\Anxiety\Test PDF"
        path1=r'D:\Msc Gogol books\ML\Summer Project\Anxiety\Test PDF\Text'
        pdftotxt(base_path,path1)
                   



###--------------Function to remove specific words from text---------------------------#####
'''
This function removes particular words from text if required.
The words have to be specified in the regular expression.
'''
def removeword(text):
    p=re.compile("abstract\n|results.\n\n?|conclusions.\n\n?|results:|conclusions:|background.\n\n?|design and methods.\n\n?|background and significance\n?|background:|aim.\n|aim:|background\n|introduction\n|article\n|keywords\n",re.DOTALL|re.I)
    text=re.sub(p,"",text)
#    print(text)
    return text

'''
Trying for named entity recognition in the text. Not used in actual framework.
'''
def sentence_postag(sentence,pat):
    st = StanfordNERTagger('D:/Msc Gogol books/ML/stanford-ner-2018-02-27/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
					   'D:/Msc Gogol books/ML/stanford-ner-2018-02-27/stanford-ner-2018-02-27/stanford-ner.jar',
					   encoding='utf-8')
    sent=word_tokenize(sentence)
    tagged_text = st.tag(sent)
    print(tagged_text)
 

'''
Deletes files in the folder passed as parameter.
'''
def deletefiles(folder):
    for filename in os.listdir(folder):
        path=join(folder,filename)
        if isfile(path):
            os.remove(path)

'''
Move all files from one folder location to another.
Both paths passed as parameter.
'''
def movefiles(source_folder,dest_folder,exception_file):
    for filename in os.listdir(source_folder):
        if (filename is not exception_file):
            source_path=join(source_folder,filename)
            dest_path=join(dest_folder,filename)
            if isfile(source_path):
                os.rename(source_path,dest_path)

####---------For checking if sentence is Table or not.---------------------#####
'''
This function checks if the text passed as parameter is a table text or not.
'''
def table_extract(text):
#    sent=sent_tokenize(text) 
    p1=re.compile("Table.*\n.*\n|TABLE.*\n.*\n")
#    for i in sent:
    if p1.search(text):
        return 1
    else:
        return 0


####-------------Function for creating positive and negative files--------------------#####
'''
This function divides the text in the file as anxiety ( positive) or non-anxiety() data.
First it is checked if the sentence is a Table sentence or not.
    If it is a table sentence and contains the keywords mentioned in p4 expression, then anxiety.
    Else if the sentence does not contain 'anxiety' then non-anxiety.
        Else discard the sentence.
    Else if the sentence contains keywords, digits and 'SD' ( expressions p4, p2, p3), then anxiety
    Else if the sentence does not contain 'anxiety, then non-anxiety
        Else discard the sentence
The sentences thus classified are kept in a single file as per anxiety or non-anxiety and
then sent to the respective folder.
'''
def posneg(data,filename,base_pos,base_neg,type_data):
        
        p4=re.compile('STAI|S-STAI|STAIY-1|SAI|STATE_TRAIT|BAI Beck Anxiety Inventory|STATE.*ANXIETY|TRAIT.*ANXIETY|ANXIETY.*LEVEL|State-Trait Anxiety Inventory|anxiety.*state|anxiety.*trait|level.*anxiety',re.I)

        p2=re.compile('\d+')
        p3=re.compile(' SD ')
        p5=re.compile("anxiety",re.I)
       
#        print (type(data))
        if type_data == "train":
            sent=sent_tokenize(data) 
    #        pos=[]
    #        neg=[]
            poscount=0
            negcount=0
      #      table=[]
            for i in sent:
                if (table_extract(i))==1:
                    tab=i.split("\n")
                    for j in tab:
                        if p4.search(j):
                            #pos.append(j)
                            j=removeword(j)
                            path_pos=os.path.join(base_pos+"/"+str(poscount)+filename)
                            with open(path_pos, "wb") as my_log:
                                my_log.write((j.encode("utf-8")))
                                poscount+=1
                        else:
                            #neg.append(j)
                            j=removeword(j)
                            if (p5.search(j) is None):
                                path_neg=os.path.join(base_neg+"/"+str(negcount)+filename)
                                with open(path_neg, "wb") as my_log:
                                    my_log.write((j.encode("utf-8")))
                                    negcount+=1
                elif (p4.search(i) and p2.search(i) and p3.search(i)):
                    #pos.append(i)
                    i=removeword(i)
                    path_pos=os.path.join(base_pos+"/"+str(poscount)+filename)
                    with open(path_pos, "wb") as my_log:
                        my_log.write((i.encode("utf-8")))
                        poscount+=1
                else:
                    #neg.append(i)
                    i=removeword(i)
                    if (p5.search(i) is None):
                        path_neg=os.path.join(base_neg+"/"+str(negcount)+filename)
                        with open(path_neg, "wb") as my_log:
                            my_log.write((i.encode("utf-8")))
                            negcount+=1
        elif type_data == "test":
            sent=sent_tokenize(data) 
            p6=re.compile('SAI|state|trait|',re.I)
            pos=[]
            neg=[]
            for i in sent:
                if (table_extract(i))==1:
                    tab=i.split("\n")
                    for j in tab:
                        if p4.search(j) or p6.search(j):
                            pos.append(j)
                            
                        else:
                            neg.append(j)
                            
                elif (p4.search(i) and p2.search(i) and p3.search(i)):
                    pos.append(i)
                    
                else:
                    neg.append(i)
            posi="\n\n".join(pos)   
            negi="\n\n".join(neg)
            path_pos=os.path.join(base_pos+"/"+filename)
            path_neg=os.path.join(base_neg+"/"+filename)
            posi=removeword(posi)
            negi=removeword(negi)
            if posi:
                with open(path_pos, "wb") as my_log:
                            my_log.write((posi.encode("utf-8")))
                print("Done (pos)!!") 
            if negi:
                with open(path_neg, "wb") as my_log:
                            my_log.write((negi.encode("utf-8")))
                print("Done (neg)!!")                   
                            
                            

####----------------Function for extracting Abstract------------------#####
'''
Function to extract only Abstract of a file if it is present in the pattern mentioned in the expressions.
'''        
def abs_ext(f1): 
            p=re.compile("abstract.*key\s*words|abstract.*background\n|abstract.*background and significance\n|abstract.*abbreviations:|abstract.*Introduction\n",re.DOTALL|re.I)
            abstract1=(p.findall(f1))
            if abstract1:
                return abstract1
                
            else:
                p2=re.compile(".*key\s*words|.*background\n|.*abbreviations:|.*Introduction\n",re.DOTALL|re.I)
                abstract2=(p2.findall(f1))
                if abstract2:
                    return abstract2

                else: 
                    p2=re.compile("abstract.*\n",re.I)
                    abstract3=(p2.findall(f1))
                    if abstract3:
                        return abstract3


####---------------Function for taking user input for using Abstract or full PDF-----------###
'''
Function to take user input of whether only the Abstract is to be used or the entire text.
'''
def abs_ext_choice(base_path,choice1,base_pos,base_neg,type_data):    
    count=0
    print (base_path)
    for filename in os.listdir(base_path):
        print(count)
        count=count+1
        print (filename)
        if filename.endswith(".txt"):
            pathtxt=os.path.join(base_path+"/"+filename)
            f=open(pathtxt,encoding="utf-8")
            f1=f.read()
            
            if (choice1=="Y" or choice1=="y"):  
                abst=abs_ext(f1)
                if (abst!=None):
                    posneg(abst[0],filename,base_pos,base_neg,type_data)
            elif (choice1=="N" or choice1=="n"):
                f1=extract_main_text(f1)
#                print(f1)
                posneg(f1,filename,base_pos,base_neg,type_data)
            else:
                print ("invalid choice")
    
'''
This function removes the text portion above the Abstract, if present, and the References part.
'''
def extract_main_text(f1):
    p=re.compile("abstract.*references",re.DOTALL|re.I)
    abstract1=(p.findall(f1))
    if abstract1:
        print("Extracting Abstract to References")
        return abstract1[0]
        
    else:
        p2=re.compile(".*references",re.DOTALL|re.I)
        abstract2=(p2.findall(f1))
        if abstract2:
            print("Extracting starting to references")
            return abstract2[0]

        else: 
            print("Extracting full text")
            return f1


  
###-------------Function for creating classifier pipelines as per choice--------------#####
'''
Function that takes user input for choosing the classifier to be used and creates the classifier pipeline.
'''
def classifier():
    inp1 = input("Select Classifier: \n\t 1. AdaBoost with Random Forest  \n\t 2. Logistic Regression \n\t 3. Random Forest \n\t 4. SVM with Linear SVC\n")
    if inp1=='1':
        base=RandomForestClassifier(max_features=None)
        clf = AdaBoostClassifier(base)  
        parameters = {'clf__n_estimators':(100,200,500),
#                      'base__criterion':("gini","entropy"),
#                      'base__n_estimators':(50,100,200,500,1000),'base__max_depth':(10,20,50),
#                      'base__class_weight':('balanced',None),
                      }
    elif inp1=='2':
        clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
        parameters = {}
#        'clf__random_state':(0,10),
#        } 
    elif inp1=='3':
        clf = RandomForestClassifier(class_weight='balanced',n_estimators=50,criterion="gini",max_depth=10,max_features=None)
#        clf = RandomForestClassifier(max_features=None)
        parameters = {}
#        'clf__n_estimators':(30,40,50),
#        'clf__max_depth':(5,10),'clf__criterion':("gini","entropy"),
#        'clf__class_weight':('balanced',None),}  
  
    elif inp1=='4':
        list_C = np.arange(1,10,1)
        clf = svm.LinearSVC()  
        parameters = {
        'clf__C':list_C,
        }
    else:
        print('Invalid Input for Classifier Selection.')
    


    inp2=input("Select Pipeline:\n\t 1. TF-IDF Vectorize, then classify \n\t 2. Chi Square feature select, then classify\n")

    if inp2=="1":
        line=Pipeline([('vect',CountVectorizer(min_df=1,token_pattern=r'\b\w+\b',ngram_range=(1,3))),
                   ('tfidf',TfidfTransformer(use_idf=True,norm='l2')),('clf',clf),])
    elif inp2=="2":
        line= Pipeline([('feature_selection', SelectKBest(chi2, k=1900)),    
#        ('feature_selection', SelectKBest(mutual_info_classif, k=1000)),     
        ('clf', clf),])
    else:
        print("Invalid Input for Pipeline Selection.")

    return line,parameters

'''
This function takes the test file folder path, the stemmer to use and the best trained classified
as parameter and then just prints the positively predicted sentences without showing TP, FP, FN, TN values.
'''
def test_classify(test_path,porter,gs_clf_b):
    for filename in os.listdir(test_path):
                print ("------------",filename)            
                pathtxt=os.path.join(test_path+"/"+filename)
                
                if os.path.exists(pathtxt):        
                    td1 = open(pathtxt,encoding='UTF-8')
                    testdata1=td1.read()
                    testdata1=extract_main_text(testdata1)
#                    t1=testdata1.split("\n\n")
                    t1=sent_tokenize(testdata1)
                    
                    tokenized_t1 = [word_tokenize(doc.lower()) for doc in t1]
    
                    Lemmatized_t1=[]
                    for doc in tokenized_t1 :
                        temp=[]
                        for token in doc:
                            if ((token not in stopwords.words('english')) and (not token.isdigit())) :
                                if type(porter)==nltk.stem.porter.PorterStemmer:
                                    temp.append(porter.stem(token))
                                elif type(porter)==nltk.stem.wordnet.WordNetLemmatizer:
                                    temp.append(porter.lemmatize(token))
                        Lemmatized_t1.append(temp)    
                    
                    joined_t1 = []
                    s = " "
                    for i in range(0,len(Lemmatized_t1)):
                         d = s.join(Lemmatized_t1[i])
                         joined_t1.append(d)
                   
                    if joined_t1:
                        predp=gs_clf_b.predict(joined_t1)
                        count=0
                        for i_p in range(len(predp)):
                            #print (i_p)
                            if predp[i_p]==1:
                                #print("Positive: ",t1[i_p],"\n\n")
                                p3=re.compile(' SD ')
                                if p3.search(t1[i_p]):
                                    print("Positive: ",t1[i_p],"\n\n")
                                count=count+1
                        print(count)        


###------------Main classification function--------------------########
'''
This is the main classification function where the training and test data 
is called, the classifier is trained and tested and the TP, FP, FN and TN values are printed.
The above written test_classify function is also called here, if needed. Else it remains commented.
'''
def classify(train_path,test_path,test_pos,test_neg):
    print ("Enter 1 for Stemming and 2 for Lemmatizing")
    choice=input()
    if choice=="1":
        porter = PorterStemmer()
    elif choice=="2":    
        porter = WordNetLemmatizer()
    else:
        print ("wrong choice")
    traindata = load_files(train_path,load_content=True,encoding='utf-8',shuffle=False)
    tokenized_docs_train = [word_tokenize(doc.lower()) for doc in traindata.data]
#    
    Lemmatized_train=[]
    for doc in tokenized_docs_train :
        temp=[]
        for token in doc:
            if ((token not in stopwords.words('english')) and (not token.isdigit())) :
                if choice=="1":
                    temp.append(porter.stem(token))
                elif choice=="2":
                    temp.append(porter.lemmatize(token))
        Lemmatized_train.append(temp)    
    
    joined_train_lemmatized = []
    s = " "
    for i in range(0,len(Lemmatized_train)):
         d = s.join(Lemmatized_train[i])
         joined_train_lemmatized.append(d)
    
    ###------For finding size of vocabulary-----------###
#    vect = CountVectorizer(min_df=2)
#    vect.fit_transform(tokenized_docs_train)
#    #Y = pd.DataFrame(X.A, columns=vect.get_feature_names()).to_string()
#    #print(Y[ :-3])
#    print("Vocabulary size:",len(vect.vocabulary_))

    
    
    ###-----------------Calling Classifier function--------------------####
    line1,parameters=classifier()
    
    gs_clf=grid_search.GridSearchCV(line1,parameters,cv=5,scoring='f1_micro')
    gs_clf.fit(joined_train_lemmatized,traindata.target)
    gs_clf_b=gs_clf.best_estimator_
    print (gs_clf_b.steps)
    true_pos=0
    false_neg=0
    true_neg=0
    false_pos=0
#    
##    test_classify(test_path,porter,gs_clf_b)
#    
    for filename in os.listdir(test_path):
                print ("------------",filename)            
                true_pos=0
                false_neg=0
                true_neg=0
                false_pos=0
                pathtxt=os.path.join(test_pos+"/"+filename)
                pathtxt1=os.path.join(test_neg+"/"+filename)
                if os.path.exists(pathtxt):        
                    td1 = open(pathtxt,encoding='UTF-8')
                    testdata1=td1.read()
                    testdata1=extract_main_text(testdata1)
                    t1=testdata1.split("\n\n")
                    
                    tokenized_t1 = [word_tokenize(doc.lower()) for doc in t1]
    
                    Lemmatized_t1=[]
                    for doc in tokenized_t1 :
                        temp=[]
                        for token in doc:
                            if ((token not in stopwords.words('english')) and (not token.isdigit())) :
                                if choice=="1":
                                    temp.append(porter.stem(token))
                                elif choice=="2":
                                    temp.append(porter.lemmatize(token))
                        Lemmatized_t1.append(temp)    
                    
                    joined_t1 = []
                    s = " "
                    for i in range(0,len(Lemmatized_t1)):
                         d = s.join(Lemmatized_t1[i])
                         joined_t1.append(d)
                   
                    predp=gs_clf_b.predict(joined_t1)
                    for i_p in range(len(predp)):
                        #print (i_p)
                        if predp[i_p]==1:
                            
                            p3=re.compile('anxiety',re.I)
                            p4=re.compile('\d+')
                            if (p3.search(t1[i_p]) and p4.search(t1[i_p])):
                                true_pos=true_pos+1
                                print("TP: ",t1[i_p])
                            
                        else :
                            false_neg=false_neg+1
                            print("FN: ",t1[i_p])
                if os.path.exists(pathtxt1):        
                    td2 = open(pathtxt1,encoding='UTF-8')
                    testdata2=td2.read()
                    testdata2=extract_main_text(testdata2)
                    t2=testdata2.split("\n\n")
                    tokenized_t2 = [word_tokenize(doc.lower()) for doc in t2]
    
                    Lemmatized_t2=[]
                    for doc in tokenized_t2 :
                        temp=[]
                        for token in doc:
                            if ((token not in stopwords.words('english')) and (not token.isdigit())) :
                                if choice=="1":
                                    temp.append(porter.stem(token))
                                elif choice=="2":
                                    temp.append(porter.lemmatize(token))
                        Lemmatized_t2.append(temp)    
                    
                    joined_t2 = []
                    s = " "
                    for i in range(0,len(Lemmatized_t2)):
                         d = s.join(Lemmatized_t2[i])
                         joined_t2.append(d)
                   
                    predn=gs_clf_b.predict(joined_t2)
                    for i_n in range(len(predn)):
                        if predn[i_n]==0:
                            true_neg=true_neg+1
                            #print("TN: ",t2[i_n])
                        else :
                            
                            p3=re.compile('anxiety',re.I)
                            p4=re.compile('\d+')
                            if (p3.search(t2[i_n]) and p4.search(t2[i_n])):
                                false_pos=false_pos+1 
                                print("FP: ",t2[i_n])
                print ("tp: ",true_pos)
                print ("fn: ",false_neg)
                print ("fp: ",false_pos)
                print ("tn: ",true_neg)
                if ((true_pos+false_pos) is not 0 and (true_pos+false_pos) is not 0.0):
                    precision=(true_pos/(true_pos+false_pos))
                else:
                    precision=0    
                print ("precision:",precision)
                if ((true_pos+false_neg) is not 0 and (true_pos+false_neg) is not 0.0):
                    recall=(true_pos/(true_pos+false_neg))
                else:
                    recall=0
                print ("recall:",recall)
                
                if ((recall+precision) != 0 and (recall+precision) !=  0.0):
                    f1=((2*recall*precision)/(recall+precision))
                else:
                    f1=0
                print("f1:",f1)
                print(true_pos,false_neg,false_pos,true_neg,recall,precision)
                print("----------------")

    


####------------Main controlling part, user interface-------------------#####
'''
This is main controller part of the program. The user inputs asked here are present with proper captions.
The folder paths will need to be changed depending on the machine on which this is run.
'''                

print("Do you want to use any previously tested file as a training file? Y/N\n\n")
f_choice=input()
if (f_choice=="Y" or f_choice=="y"):
    print("Enter current test file name, not to be moved:\n")
    filen=input()
    source_folder=r'./Test PDF'
    dest_folder=r'./Train PDF'
    movefiles(source_folder,dest_folder,filen)
else:
    print("Using all existing files in test folder as test files. Moving on.\n\n")


print ("Convert pdf files to text files? Y/N")
main_choice=input()
if (main_choice=="Y" or main_choice=="y"):
    pdftotxtchoice()


print ("Create sentence wise positive negetive files? Y/N")
choice=input()
if (choice=="Y" or choice=="y"): 
    basetrain_path1=r'./Train PDF/Text'  

    
    basetrain_pos_abs= r'./Train Abstract Text/Positive'
    basetrain_neg_abs=r'./Train Abstract Text/Negative'
    
    basetrain_pos_full= r'./Train Full PDF Text/Positive'
    basetrain_neg_full=r'./Train Full PDF Text/Negative' 
    
    print("Extract only abstract from training set? Y/N, enter any key for skipping this step.")
    choice1=input()
    
    if (choice1=="Y" or choice1=="y"):
        print("Delete any existing positive and negative file?Y/N \n\n")
        choice15=input()
            
        if((choice15=="Y") or (choice15=="y")):
            deletefiles(basetrain_pos_abs)
            deletefiles(basetrain_neg_abs)
        else:
            print("Not deleting files. Moving on.")
        print("For anxiety files")
        abs_ext_choice(basetrain_path1,choice1,basetrain_pos_abs,basetrain_neg_abs,"train")

    elif (choice1=="N" or choice1=="n"):
        print("Delete any existing positive and negative file?Y/N \n\n")
        choice16=input()
            
        if((choice16=="Y") or (choice16=="y")):
            deletefiles(basetrain_pos_full)
            deletefiles(basetrain_neg_full)
        else:
            print("Not deleting files. Moving on.")
        print("For anxiety files")
        abs_ext_choice(basetrain_path1,choice1,basetrain_pos_full,basetrain_neg_full,"train")

    else:
        print("Skipping this step.\n")

    
    basetest_path1=r'./Test PDF/Text'


    basetest_pos_abs= r'./Test Abstract Text/Positive'
    basetest_neg_abs=r'./Test Abstract Text/Negative'
    
    basetest_pos_full= r'./Test Full PDF Text/Positive'
    basetest_neg_full=r'./Test Full PDF Text/Negative'
    
    print("Extract only abstract from test set? Y/N, enter any key for skipping this step.")
    choice2=input()
    
    if (choice2=="Y" or choice2=="y"):
        print("Delete any existing positive and negative file?Y/N \n\n")
        choice17=input()
            
        if((choice17=="Y") or (choice17=="y")):
            deletefiles(basetest_pos_abs)
            deletefiles(basetest_neg_abs)
        else:
            print("Not deleting files. Moving on.")
        print("For test files")
        abs_ext_choice(basetest_path1,choice2,basetest_pos_abs,basetest_neg_abs,"test")
#       
    elif (choice2=="N" or choice2=="n"):
        print("Delete any existing positive and negative file?Y/N \n\n")
        choice18=input()
            
        if((choice18=="Y") or (choice18=="y")):
            deletefiles(basetest_pos_full)
            deletefiles(basetest_neg_full)
        else:
            print("Not deleting files. Moving on.")
        print("For test files")
        abs_ext_choice(basetest_path1,choice2,basetest_pos_full,basetest_neg_full,"test")

    else:
        print("Skipping this step.\n")

else:
    print ("Moving to the next step: Classification")    

     
print ("Enter 1 to train using only abstract. Enter 2 to train using full pdf")   
choicetr=input()
print ("Enter 1 to test using only abstract. Enter 2 to test using full pdf")
choicets=input()
if choicetr=="1":
    basepath1_train=r'./Train Abstract Text'
elif choicetr=="2":    
    basepath1_train=r'./Train Full PDF Text'  
else:
    print ("Invalid choice for classification training.")                      
    
basepath1_test=r'./Test PDF/Text'

if choicets=="1":
    path1_test=(r'./Test Abstract Text/Positive')
    path2_test=(r'./Test Abstract Text/Negative')
elif choicets=="2":    
    path1_test=(r'./Test Full PDF Text/Positive')
    path2_test=(r'./Test Full PDF Text/Negative')
else:
    print ("Invalid choice for classification test.")       

classify(basepath1_train,basepath1_test,path1_test,path2_test)        
    
    
    
#----------------Printing execution time-------------------------------------
print("--- %s seconds ---" % (time.time() - start_time))
    
    
