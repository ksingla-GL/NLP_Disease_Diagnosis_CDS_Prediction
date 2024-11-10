# NLP_Disease_Diagnosis_CDS_Prediction

Data Required - For now demo data is provided as Symptoms-Diagnosis.txt
Enviroment and Packages required - requirements.txt
Entire Pipeline Code - CDS_preds.py

# Continued Optimization Plan 
Understanding and Reusing Cython Code: The original Cython code is extensive and lacks structured documentation, making it challenging to follow. However, investing time to dissect this code, identify its core logic, and adapt it into a streamlined pipeline could improve our results. By leveraging more of the original code’s optimizations, we could achieve better alignment with the authors' methodology and potentially boost accuracy. 

Using Larger Datasets: Scaling up the dataset for training and testing could lead to more reliable and generalizable results. The current experiments rely on a limited subset, but processing additional data would help capture a broader range of patient profiles and symptom variations, enhancing model performance. 

Experimenting with sent2vec and Higher GPUs: While using sent2vec was initially infeasible due to dependency issues and computational costs, we could revisit this option if we secure access to high-performance GPUs. Training sent2vec with medical-specific corpora on larger GPUs could align our model more closely with the original’s intent, potentially increasing both accuracy and precision. 

Resolving Package Issues: The complex dependencies associated with sent2vec and other legacy libraries like gensim presented obstacles. Overcoming these compatibility issues, possibly through environment isolation or by modifying source code, could allow us to experiment with the original architecture more effectively. 

Exploring Additional Embedding Models: While SentenceTransformers proved effective, experimenting with alternative embedding models, particularly those trained on biomedical text (such as BioBERT or SciBERT), could provide further improvements. These models would enhance the semantic understanding of medical terminologies, leading to potentially higher precision in diagnosis prediction. 
