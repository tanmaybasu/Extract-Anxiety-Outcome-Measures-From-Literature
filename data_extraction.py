#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday, December 22, 2020 @ 19:45:28
@author: Tanmay Basu
"""

import csv,os,re,sys
import fitz
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif 


class data_extraction():
     def __init__(self,path='/home/data_extrcation/'):
        self.path = path

# PDF to text conversion
     def pdf_to_text(self,file):        
        try:
            doc = fitz.open(file, filetype = "pdf")        # PDF to text conversion  
            texts=[] 
            for page in doc:
                texts.append(page.getText().encode("utf8"))                   
            try:
                text=b''.join(texts)                      # If Fitz return the text in binary mode 
                text=text.decode('utf-8')                   
            except:
                text=''.join(texts)
        except:
            text='' 
        return text

# Text refinement
     def text_refinement(self,text='hello'):
        text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+',' ', text)      # Remove special characters e.g., emoticons-😄. The ^ in the beginning ensures that all the natural characters will be kept. 
        text=re.sub(r'[?]', '.', text)                                             # Replace '?' with '.' to properly identify floating point numbers 
        text=re.sub(r'([a-zA-Z0-9])([\),.!?;-]+)([a-zA-Z])', r'\1\2 \3', text )    # Space between delimmiter and letter   
        return text
           
# Check if a sentence is relevant to a keyphrase 
     def check_relevant_sentences(self,sentence,keyphrases):
        sentence = re.sub(r'[^a-zA-Z0-9.?:!$\n]', ' ', sentence)        # Remove special character 
        for phrase in keyphrases:
            if sentence.find(phrase)>=0 and re.findall(r'\d+\.\d+', sentence)!=[]:
                return 1
        return 0
    
# Building training corpus  
     def build_training_data(self):        
         if os.path.isfile(self.path+'keyphrases.txt') and os.path.getsize(self.path+'keyphrases.txt') > 0:                      
             fk=open(self.path+'keyphrases.txt', "r") 
             keyphrases = list(csv.reader(fk,delimiter='\n'))
             keyphrases = [item for sublist in keyphrases for item in sublist]
             fk.close() 
         else:                                                  # If the keyphrases file does not exist or is empty then exit 
             print('Either keyphrases.txt does not exist or it is empty \n')
             sys.exit(0)           
    # If the training data directory does not exist then exit
         if not os.path.isdir(self.path+'training_data'):               
            print('The directory training_data does not exist \n')
            sys.exit(0)
         trn_files=os.listdir(self.path+'training_data')
         if trn_files==[]:
            print('There is no training samples in the directory \n')
         else:
            count=0;
            print('############### Preparing Training Data ############### \n')
            fp=open(self.path+'training_relevant_class_data.csv',"w")
            fn=open(self.path+'training_irrelevant_class_data.csv',"w")  
            for item in trn_files:
                count+=1
                text=self.pdf_to_text(self.path+'training_data/'+item)      # PDF to text conversion 
                text=self.text_refinement(text)                             # Cleaning text file 
                fp.write('trn '+str(count)+',')
                fn.write(str(count)+',')
                text=re.sub(r',', r';', text)      # replacing , by ; to build the csv properly 
                text=re.sub(r'\n', r' ', text)     # replacing \n by ; to build the csv properly
                if text!='':
                    sentences=nltk.sent_tokenize(text)
                    for sentence in sentences:
                        if sentence!='':
                            phrase_score=self.check_relevant_sentences(sentence,keyphrases)
                            if phrase_score==1:
                                fp.write(sentence+' ')
                            elif phrase_score==0 and re.findall(r'\d+', sentence)==[]:
                                fn.write(sentence+' ') 
                    fp.write('\n')
                    fn.write('\n')
                else:
                    print('Empty text for file: '+item+'\n') 
         fp.close()
         fn.close()
        
# Selection of classifiers and building classification pipeline 
     def classification_pipeline(self,opt,no_term,trn_data,trn_cat):        
        # Logistic Regression 
        if opt=='lr':
            print('\n\t### Classification of given texts using Logistic Regression Classifier ### \n')
            ext2='logistic_regression'
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
        # Linear SVC 
        elif opt=='ls':   
            print('\n\t### Classification of given texts using Linear SVC Classifier ### \n')
            ext2='linear_svc'
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,2,10,50,100),
            }         
        # Multinomial Naive Bayes
        elif opt=='n':
            print('\n\t### Classification of given texts using Multinomial Naive Bayes Classifier ### \n')
            ext2='naive_bayes'
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }            
        # Random Forest 
        elif opt=='r':
            print('\n\t ### Classification of given texts using Random Forest Classifier ### \n')
            ext2='random_forest'
            clf = RandomForestClassifier(criterion='gini',max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__n_estimators':(30,50,100,200),
            'clf__max_depth':(10,20),
            }          
        # Support Vector Machine 
        elif opt=='s': 
            print('\n\t### Classification of given texts using Linear SVM Classifier ### \n')
            ext2='svm'
            clf = svm.SVC(kernel='linear', class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,0.5,1,2,10,50,100),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)
    # Classificiation and feature selection pipelines
        if no_term==0:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\b\w+\b',stop_words=stopwords.words('english'))),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
                ('clf', clf),]) 
        else:
            try:                                        # To use selected terms of the vocabulary
                pipeline = Pipeline([
                    ('vect', CountVectorizer(token_pattern=r'\b\w+\b',stop_words=stopwords.words('english'))),
                    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                    ('feature_selection', SelectKBest(chi2, k=no_term)),    
            #        ('feature_selection', SelectKBest(mutual_info_classif, k=no_term)),        
                    ('clf', clf),]) 
            except:                                  # If the input is wrong
                print('Wrong Input. Enter number of terms correctly. \n')
                sys.exit()
    # Fix the values of the parameters using Grid Search and cross validation on the training samples 
        feature_parameters = {
        'vect__min_df': (2,3),
        'vect__ngram_range': ((1, 2),(1,3)),  # Unigrams, Bigrams or Trigrams
        }
        parameters={**feature_parameters,**clf_parameters}
    
        grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_
#        print(grid.best_params_)
        return clf,ext2
    
# Classification of sentences containing desired data elements   
     def data_extraction(self):
        opt = input("Choose a classifier : \n\n\t 'lr' to select Logistic Regression" 
                       "\n\t 'r' to select Random Forest" 
                       "\n\t 'ls' to select Linear SVC" 
                       "\n\t 's' to select SVM" 
                       "\n\t 'n' to select Naive Bayes \n\n")
        no_term= input("Enter: \n\t '0' to use all the terms of the vocabulary" 
                       "\n\t 'DESIRED' number of terms to choose using chi-square statistics \n\n")
        no_term=int(no_term)
        trn_data=[];    trn_cat=[];   
        p1=0; p2=0; p3=0
    
        fp=open(self.path+'training_relevant_class_data.csv',"r")
        fn=open(self.path+'training_irrelevant_class_data.csv',"r")    
        rel_sent = list(csv.reader(fp,delimiter='\n')) 
        irl_sent = list(csv.reader(fn,delimiter='\n'))
        
    # Getting Relevant Sentences
        for item in rel_sent:
            text=''.join(item)
            if text.split(',')[1]:
                text=text.split(',')[1]
                sentences = tokenize.sent_tokenize(text)
                for sentence in sentences:
                    sentence=re.sub(r'\d+\.\d+', '', sentence)        # Remove numbers from text to reduce text features 
                    trn_data.append(sentence)           
                    trn_cat.append(0)
                    p1=p1+1
      
    # Getting Irrelevant Sentences       
        for item in irl_sent:
            text=''.join(item)
            if text.split(',')[1]:
                text=text.split(',')[1]
                sentences = tokenize.sent_tokenize(text)
                for sentence in sentences:
                    sentence=re.sub(r'\d+\.\d+', '', sentence)        # Remove numbers from text to reduce text features 
                    trn_data.append(sentence)           
                    trn_cat.append(1)
                    p2=p2+1
    # Classification pipeline                   
        clf,ext2=self.classification_pipeline(opt,no_term,trn_data,trn_cat)   
    # Classification of the test samples  
        if not os.path.isdir(self.path+'test_data'):
            print('The directory test_data does not exist \n')
            sys.exit(0)
        else:
            tst_files=os.listdir(self.path+'test_data')
        p3=0; count=0;
        if tst_files==[]:
            print('There is no test samples in the directory \n')
        else:
            for item in tst_files:
                count+=1
                tst_data=[];   
                text=self.pdf_to_text(self.path+'test_data/'+item)   
                out = open(self.path+'output/tst'+str(count)+'.txt',"w") 
                # Preparing Test Samples 
                if text!='':
                    text=self.text_refinement(text)                   # Cleaning text file 
                    sentences = tokenize.sent_tokenize(text)
                    for sentence in sentences:                       # Extracting sentences from text
                        tst_data.append(sentence)  
                        p3=p3+1
                # Cretaing the output file
                    out.write('\n Using '+ext2+' Classifier: \n\n')   
                    out.write('Total No. of Sentences in Test Sample: '+str(p3)+'\n\n')
                    out.write('The relevant sentences are as follow: \n')
                # Classification
                    predicted = clf.predict(tst_data) 
                    nps=0
                    for i in range(0,len(predicted)):
                        if predicted[i] == 0: 
                                nps=nps+1                 
                                out.write('\n'+str(nps)+")  "+tst_data[i]+'\n')               
                    print("Total No. of Relevant Sentences of Test Sample "+str(count)+" : %d\n" %nps)
                    out.close()
                else:
                    out.write('Test file '+str(count)+'is empty \n') 
                    print('Test file '+str(count)+' is empty \n')
                out.close()
            print('No of sentences belong to RELEVANT class of the training corpus: '+ str(p1)) 
            print('No of sentences belong to IRRELEVANT class of the training corpus: '+ str(p2)) 
            print('No of sentences belong to the TEST corpus: '+ str(p3)) 
    

