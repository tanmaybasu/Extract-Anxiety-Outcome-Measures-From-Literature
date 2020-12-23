#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday, December 22, 2020 @ 15:30:18

@author: Tanmay Basu 
"""

import os
import re,csv 
import fitz
import nltk
from nltk.corpus import stopwords


class build_training_data():
    def __init__(self,data_path='/home/data_extrcation/'):
        self.path = data_path

    # Text refinement
    def text_refinement(self,text='hello'):
        text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+',' ', text)      # Remove special characters e.g., emoticons-😄. The ^ in the beginning ensures that all the natural characters will be kept. 
#        text = re.sub(r'[^a-zA-Z0-9.?:!$\n]', ' ', text)                          # Remove special characters
        text=re.sub(r'[?]', '.', text)                                             # Replace '?' with '.' to properly identify floating point numbers 
        text=re.sub(r'([a-zA-Z0-9])([\),.!?;-]+)([a-zA-Z])', r'\1\2 \3', text )    # Space between delimmiter and letter   
        text=re.sub(r'([a-z])([\.])([\s]*)([a-z])', r'\1 \3\4', text)              # Reomove '.' between two lowercase letters e.g., et al. xxx
        text=re.sub(r'([a-z])([\.]*)([0-9])', r'\1\2 \3', text)                    # Space between letter and no.    
        text=re.sub(r'(\s)([a-z0-9]+)([A-Z])([\w]+)', r'\1\2. \3\4', text)         # Put a '.' after a lowercase letter/number followed by Uppercase e.g., drains removed by day threeHe continued to 
        text=re.sub(r'(\.)([\s]*)([\.]+)', r'\1', text)                            # Removing extra '.'s, if any   

        return text
    
    # Longest Common Subsequence
    def lcs(self,s1, s2):
        m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
        longest = 0
        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                else:
                    m[x][y] = 0
        return longest
        
    # Final score calculation
    def get_sent_score(self,sent,phrase,inp):
        sent_tokens = nltk.word_tokenize(sent.lower().strip('.'))
        sent_tokens = [item.rstrip('s') for item in sent_tokens]        # Converting the plurals to singulars   
        target_tokens = nltk.word_tokenize(phrase.lower().strip('.'))              
        target_tokens = [item.rstrip('s') for item in target_tokens]    # Converting the plurals to singulars             
        score = 0   
    # Using modified Jaccard Similarity    
        if inp=='0':                                                    # Jaccard is the default distance  
            for token in target_tokens:
                if token in sent_tokens and token not in stopwords.words('english'):    # Discarding the stopwords 
                    score += 1
            if score!=0:
                score = float(score)/len(target_tokens)   
    # Using LCS     
        else:
            score=self.lcs(self,sent_tokens,target_tokens)
            if score>0:
                score=score/float(len(target_tokens))     
        return score
    
    # check if a sentence is relevant to the data element 
    def check_relevant_sentences(self,sentence,keyphrases):
        phrase_score=[]; total_score=0.0
        sentence = re.sub(r'[^a-zA-Z0-9.?:!$\n]', ' ', sentence)    # Remove special character 
        for phrase in keyphrases:
            score=0.0;
            score=self.get_sent_score(sentence,phrase,'0')
            total_score+=score
            if score>=0.5:
                tmp=[]
                tmp.append(phrase)
                tmp.append(score)
                phrase_score.append(tmp)
        if phrase_score!=[]:
            phrase_score.sort(key=lambda x: x[1], reverse=True)     # Sorting the phrases according to ascending order of similarity scores 
            return phrase_score
        if total_score==0:
            tmp=[]; 
            tmp.append(phrase)
            tmp.append(score)
            phrase_score.append(tmp)
        return phrase_score
    
    # Building training corpus  
    def build_training_data(self):
        os.system('ls '+self.path+'training_data/*.pdf> '+self.path+'list_training_data.txt')
        fl=open(self.path+'list_training_data.txt', "r") 
        trn_files = list(csv.reader(fl,delimiter='\n')) 
        trn_files=[item for sublist in trn_files for item in sublist]
        fl.close()
    
        fk=open(self.path+'keyphrases.txt', "r") 
        keyphrases = list(csv.reader(fk,delimiter='\n'))
        keyphrases = [item for sublist in keyphrases for item in sublist]
        fk.close() 
        count=0;
        print('############### Preparing Training Data ############### \n')
        fp=open(self.path+'training_relevant_class_data.csv',"w")
        fn=open(self.path+'training_irrelevant_class_data.csv',"w")  
        for item in trn_files:
            count+=1
            try:
              doc = fitz.open(item, filetype = "pdf")        # PDF to text conversion  
              texts=[] 
              for page in doc:
                 texts.append(page.getText().encode("utf8"))    
              try:
                  text=b''.join(texts)                       # If Fitz return the text in binary mode 
                  text=text.decode('utf-8')
              except:
                  text=''.join(texts)
            except:
              text=''                  
            text=self.text_refinement(text)                     # Cleaning text file 
            fp.write('trn '+str(count)+',')
            fn.write(str(count)+',')
            text=re.sub(r',', r';', text)      # replacing , by ; to build the csv properly 
            text=re.sub(r'\n', r' ', text)    # replacing \n by ; to build the csv properly
            sentences=nltk.sent_tokenize(text)
            for sentence in sentences:
                if sentence!='':
                    phrase_score=self.check_relevant_sentences(sentence,keyphrases)
                    if phrase_score!=[]:
                        ln=len(nltk.word_tokenize(phrase_score[0][0]))
                        if ln>2 and phrase_score[0][1]>=0.5 and re.findall(r'\d+\.\d+', sentence)!=[]:    # Check if there is a floating point number
                            fp.write(sentence+' ')
                        elif ln<=2 and phrase_score[0][1]>0.5 and re.findall(r'\d+\.\d+', sentence)!=[]:  # Check if there is a floating point number
                            fp.write(sentence+' ')
                        elif phrase_score[0][1]==0 and re.findall(r'\d+\.\d+', sentence)==[]:  # Check if there is no floating point number                           
                            fn.write(sentence+' ') 
            fp.write('\n')
            fn.write('\n')
        fp.close()
        fn.close()

 
