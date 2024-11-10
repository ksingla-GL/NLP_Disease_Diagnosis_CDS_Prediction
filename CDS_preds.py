import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import cosine
import random

import os
import shutil
import sys
from math import floor

from gensim.models import fasttext

import numpy
import numpy as np
import sent2vec
import sklearn
from scipy import spatial
from sklearn.model_selection import train_test_split, KFold

""""""
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from pathlib import Path
import matplotlib.pyplot as plt

class SymptomsDiagnosis:
    CONST_HADM_ID = 0
    CONST_SUBJECT_ID = 1
    CONST_ADMITTIME = 2
    CONST_DISCHTIME = 3
    CONST_SYMPTOMS = 4
    CONST_DIAGNOSIS = 5

    def __init__(self, hadm_id, subject_id, admittime, dischtime, symptoms, diagnosis):
        self.hadm_id = hadm_id
        self.subject_id = subject_id
        self.admittime = admittime
        self.dischtime = dischtime
        self.symptoms = symptoms
        self.diagnosis = diagnosis
        
def preprocess_diagnosis(diagnosis):
    return diagnosis.lower()

# Compute embedding for symptoms
def embending_symptoms(model, admissions):
    embendings = {}
    for hadm_id, admission in admissions.items():
        symptoms = admission.symptoms  # Direct attribute access
        embendings[hadm_id] = model.encode(symptoms)  # Encoding using the model
    return embendings

# Compute embedding for diagnosis
def embending_diagnosis(model, admissions):
    embendings = {}
    for hadm_id, admission in admissions.items():
        diagnosis = admission.diagnosis  # Direct attribute access
        embendings[hadm_id] = model.encode(diagnosis)  # Encoding using the model
    return embendings

################################################################################################################
#READ DATASET
################################################################################################################
#os.chdir(CH_DIR) # to change current working dir
file_name = os.getcwd() + "/Symptoms-Diagnosis.txt"
f = open(file_name, "r").readlines()
orig_stdout = sys.stdout

admissions = dict()
for line in f:
    line.replace("\n", "")
    attributes = line.split(';')
    a = SymptomsDiagnosis(attributes[SymptomsDiagnosis.CONST_HADM_ID], attributes[SymptomsDiagnosis.CONST_SUBJECT_ID], attributes[SymptomsDiagnosis.CONST_ADMITTIME],
                                                   attributes[SymptomsDiagnosis.CONST_DISCHTIME], attributes[SymptomsDiagnosis.CONST_SYMPTOMS],
                                                   preprocess_diagnosis(attributes[SymptomsDiagnosis.CONST_DIAGNOSIS]))
    admissions.update({attributes[SymptomsDiagnosis.CONST_HADM_ID]:a})

################################################################################################################
#LOAD MODEL
################################################################################################################
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a lightweight model
################################################################################################################
#COMPUTE EMBENDINGS
################################################################################################################
embendings_symptoms = embending_symptoms(model,admissions)
embendings_diagnosis = embending_diagnosis(model,admissions)

# Constants for evaluation
TOP_K_VALUES = [5, 10, 15, 20, 25, 30]
SYMPTOMS_SIMILARITY_THRESHOLD = 0.7
DIAGNOSIS_SIMILARITY_THRESHOLD = 0.5

# Function to split admissions and embeddings into train and test sets (80:20)
def train_test_split_admissions(admissions, embending_symptoms, embending_diagnosis, test_ratio=0.2):
    admission_ids = list(admissions.keys())
    random.shuffle(admission_ids)
    split_index = int(len(admission_ids) * (1 - test_ratio))

    train_ids = admission_ids[:split_index]
    test_ids = admission_ids[split_index:]

    train_admissions = {id_: admissions[id_] for id_ in train_ids}
    test_admissions = {id_: admissions[id_] for id_ in test_ids}
    train_embending_symptoms = {id_: embending_symptoms[id_] for id_ in train_ids}
    test_embending_symptoms = {id_: embending_symptoms[id_] for id_ in test_ids}
    train_embending_diagnosis = {id_: embending_diagnosis[id_] for id_ in train_ids}
    test_embending_diagnosis = {id_: embending_diagnosis[id_] for id_ in test_ids}

    return train_admissions, test_admissions, train_embending_symptoms, test_embending_symptoms, train_embending_diagnosis, test_embending_diagnosis

# Calculate cosine similarity between two vectors
def calculate_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def display_text_plot(data, metric_name):
    max_value = max(data.values())
    print(f"\n{metric_name} vs Top K\n" + "-" * 30)
    for k, value in data.items():
        bar = '#' * int((value / max_value) * 40)  # Scale to 40 characters
        print(f"Top {k}: {bar} ({value:.2f})")

# Evaluate the model for each fold's test data against all training data
def evaluate_model(train_admissions, test_admissions, train_embending_symptoms, test_embending_symptoms, train_embending_diagnosis, test_embending_diagnosis, top_k_values=TOP_K_VALUES):
    precision_scores, recall_scores, f1_scores = {}, {}, {}
    for k in top_k_values:
        precision_scores[k] = []
        recall_scores[k] = []
        f1_scores[k] = []

    # Loop through each test admission
    for test_id, test_admission in test_admissions.items():
        test_symptom_embed = test_embending_symptoms[test_id]
        test_diag_embed = test_embending_diagnosis[test_id]

        # Compute similarities with all train admissions
        similarity_scores = []
        for train_id, train_admission in train_admissions.items():
            train_symptom_embed = train_embending_symptoms[train_id]
            train_diag_embed = train_embending_diagnosis[train_id]

            # Calculate average similarity between symptoms and diagnoses
            symptom_similarity = calculate_similarity(test_symptom_embed, train_symptom_embed)
            diag_similarity = calculate_similarity(train_diag_embed, test_diag_embed)
            
            # Check if similarity scores pass the threshold
            #if symptom_similarity >= SYMPTOMS_SIMILARITY_THRESHOLD and diag_similarity >= DIAGNOSIS_SIMILARITY_THRESHOLD:
            similarity_scores.append((train_id, symptom_similarity,diag_similarity))
        # Sort scores and select top-k predictions
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        for k in top_k_values:
            top_k_predictions = [x[2] for x in similarity_scores[:k]]
            true_positive = len([i for i in top_k_predictions if i>DIAGNOSIS_SIMILARITY_THRESHOLD])
            fn = [x[2] for x in similarity_scores[k:]]
            fn = len([i for i in fn if i>DIAGNOSIS_SIMILARITY_THRESHOLD])
            
            # Calculate precision, recall, and F1
            precision = true_positive/k
            recall = true_positive /(fn+true_positive+0.000000001)  
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores[k].append(precision)
            recall_scores[k].append(recall)
            f1_scores[k].append(f1)

    # Print the average metrics for each top-k
    for k in top_k_values:
        avg_precision = np.mean(precision_scores[k])
        avg_recall = np.mean(recall_scores[k])
        avg_f1 = np.mean(f1_scores[k])
        
        print(f"Top-{k} Results:")
        print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1:.4f}")
        print("-" * 40)
        
    

    # Sample data
    top_k_values = [5, 10, 15, 20, 25, 30]
    precision_scores = {k: 0.1 * (i + 1) for i, k in enumerate(top_k_values)}
    recall_scores = {k: 0.08 * (i + 1) for i, k in enumerate(top_k_values)}
    f1_scores = {k: 0.09 * (i + 1) for i, k in enumerate(top_k_values)}
        
    plt.figure(figsize=(8, 6))
    plt.plot(top_k_values, precision_scores, marker='o', label="Precision")
    plt.xlabel("Top-K")
    plt.ylabel("Precision")
    plt.title("Precision vs Top-K")
    plt.grid()
    plt.show()

    # Plot Recall vs Top-K
    plt.figure(figsize=(8, 6))
    plt.plot(top_k_values, recall_scores, marker='o', label="Recall", color='orange')
    plt.xlabel("Top-K")
    plt.ylabel("Recall")
    plt.title("Recall vs Top-K")
    plt.grid()
    plt.show()
    
    # Plot F1 Score vs Top-K
    plt.figure(figsize=(8, 6))
    plt.plot(top_k_values, f1_scores, marker='o', label="F1 Score", color='green')
    plt.xlabel("Top-K")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Top-K")
    plt.grid()
    plt.show()

# Main code to split data and run evaluation
# Assuming admissions, embending_symptoms, and embending_diagnosis are already provided
train_admissions, test_admissions, train_embending_symptoms, test_embending_symptoms, train_embending_diagnosis, test_embending_diagnosis = train_test_split_admissions(admissions, embendings_symptoms, embendings_diagnosis)

# Run the evaluation
evaluate_model(train_admissions, test_admissions, train_embending_symptoms, test_embending_symptoms, train_embending_diagnosis, test_embending_diagnosis, TOP_K_VALUES)

