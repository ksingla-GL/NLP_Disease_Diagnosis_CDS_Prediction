# NLP_Disease_Diagnosis_CDS_Prediction

Data Required - Symptoms-Diagnosis.txt
Enviroment and Packages required - requirements.txt
Entire Pipeline Code Initial Draft - CDS_preds.py
Entire Pipeline Code Final- BD4H_Final.ipynb

# Code & Methodology Detailed Implementation 
1. Collect the symptoms data of patients from MIMIC 3 – split into train, test datasets 
2. Load the NLP embedding model – fine tune for general MIMIC Data
3. Embed the symptoms in training set for patients’ symptom similarities
4. Use k-fold cross validation on train data to determine optimal similarity threshold
5. Looping through each test data point, find subset of similar patients by filtering for similarity scores higher than the threshold above, ranking them in order of scores
6. For each of k=[5,10,15,20,25,30] - find the top k such similar patients and predict the final diagnosis to be the same as those for these k patients
7. Compute precision, recall & F1 scores based on diagnoses similarity scores of these k predictions for whole test set and plot the graphs 
