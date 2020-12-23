# Data Elements Extraction of Anxiety Outcome Measures From Relevant Literature using Machine Learning and NLP
The aim of this project is to build a framework using machine learning to extract required data elements of anxiety outcome measures from relevant literaure. The framework builds a training corpus by extracting senetnces containing different data elements of anxiety outcome measures from relevant publications by using some keyphrases fixed by the domain expert. The publications are retrieved from Medline, EMBASE, CINAHL, AHMED and Pyscinfo following a given set of rules by the domain expert. Subsequently, the method trains a machine learning classifier e.g., Random Forest using this training corpus to extract the sentences containing desired data elements from test samples. The experiments are conducted on 48 publications containing anxiety outcome measures with an aim to automatically extract the sentences stating the mean and standard deviation of the measures of outcomes of different types of interventions to lessen anxiety. The experimental results show that the recall and precision of the proposed method using random forest classifier are respectively 100% and 83%, which indicates that the method is able to extract all required data elements. The analysis and performance of this framework is explained in this paper:

[Shubhaditya Goswami, Sukanya Pal, Simon Goldsworthy and Tanmay Basu. An Effective Machine Learning Framework for Data Elements Extraction from the Literature of Anxiety Outcome Measures to Build Systematic Review. In Proceedings of International Conference on Business Information Systems, pp 247-258, 2019](https://link.springer.com/chapter/10.1007/978-3-030-20485-3_19).


## How to run the framework?

The training data to train the classifier can be developed by running 'build_training_data.py' using PDF files of the publications. Pass the path of the project e.g., `/home/xyz/data_extraction/` as a parameter of this function. Create the following directories inside this path: 1) `training_data`, 2) `test_data`. Therefore store the PDFs for training and test data in the respective directories. The list of keyphrases to build the training data should be stored as `keyphrases.txt` in the main project path. Subsequently, run the following lines to build the training data.

```
btd=build_training_data('/home/xyz/data_extraction/')
btd.build_training_data()
```
The desired data elements of anxiety outcome measures can be extracted from the individual test samples by executing 'data_extraction.py'. Create a directory, called, `output` in the main project path to store the outputs of individual test samples. Therefore, run the following lines to get data elements of individual test samples. 

```
clf=data_extraction('/home/xyz/data_extraction/')
clf.classify()
```

An example code to implement the whole model is uploaded as `testing_data_extraction.py`. For any further query, you may reach out to me at welcometanmay@gmail.com
