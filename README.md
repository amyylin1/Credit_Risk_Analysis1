# Credit_Risk_Analysis1

## Overview

This exercise compares and contrasts six supervised machine-learning models:

1. Naive random oversampling (RandomOverSampler library)
2. SMOTE oversampling (SMOTE library) 
3. Undersampling (ClusterCentroids algorithm)
4. Cobmination sampling (SMOTEENN algorithm)
5. Balanced random forest classifier (biased reduction)
6. Ensemble adaboost classifier (biased reduction)


## Results

### 1. Naive random oversampling

<img width="703" alt="Screen Shot 2022-11-22 at 12 38 51 PM" src="https://user-images.githubusercontent.com/108419097/203383519-3a0b9e55-a499-4440-ade5-9a3df6957740.png">

- Balanced accuracy:  0.65
- Precision (high-/low-risk loans):  0.01/1.00
- Recall (high-/low-risk loans):  0.61/0.68


### 2. SMOTE oversampling

<img width="681" alt="Screen Shot 2022-11-22 at 12 39 36 PM" src="https://user-images.githubusercontent.com/108419097/203383667-15aa1c73-9cd3-4fb8-9af5-30a5913d2448.png">

- Balanced accuracy:  0.62
- Precision (high-/low-risk loans):  0.01/1.00
- Recall (high-/low-risk loans):  0.61/0.64

### 3. ClusterCentroids algorithm

<img width="681" alt="Screen Shot 2022-11-22 at 12 43 53 PM" src="https://user-images.githubusercontent.com/108419097/203384540-1b82f351-9265-47b0-9c5f-d6fde74ef6af.png">

- Balanced accuracy:  0.62
- Precision (high-/low-risk loans):  0.01/1.00
- Recall (high-/low-risk loans):  0.60/0.43

### 4. SMOTEENN algorithm

<img width="686" alt="Screen Shot 2022-11-22 at 12 44 49 PM" src="https://user-images.githubusercontent.com/108419097/203384707-42f6ee21-2482-42b0-ab57-59c3519f5b60.png">

- Balanced accuracy:  0.51
- Precision (high-/low-risk loans):  0.01/1.00
- Recall (high-/low-risk loans):  0.70/0.58

### 5. Balanced random forest classifier

<img width="695" alt="Screen Shot 2022-11-22 at 12 45 24 PM" src="https://user-images.githubusercontent.com/108419097/203384806-173f34b1-ed44-4239-8846-171cd4a5dc83.png">

- Balanced accuracy:  0.79
- Precision (high-/low-risk loans):  0.04/1.00
- Recall (high-/low-risk loans):  0.67/0.91

### 6. Ensemble adaboost classifier

<img width="676" alt="Screen Shot 2022-11-22 at 12 45 54 PM" src="https://user-images.githubusercontent.com/108419097/203384886-706c96a2-3702-4ab6-940b-fb5efc04d27d.png">

- Balanced accuracy:  0.93
- Precision (high-/low-risk loans):  0.07/1.00
- Recall (high-/low-risk loans):  0.91/0.94

### Codes 
1. Naive random oversampling

        # Resample the training data with the RandomOversampler
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=1)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        Counter(y_resampled)

2. SMOTE oversampling

        # Resample the training data with SMOTE
        from imblearn.over_sampling import SMOTE
        X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(
            X_train, y_train)
        Counter(y_resampled)

3. Undersampling

        # Resample the data using the ClusterCentroids resampler
        from imblearn.under_sampling import ClusterCentroids
        cc = ClusterCentroids(random_state=1)
        X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
        Counter(y_resampled)

4.  Cobmination (over and under) sampling

        # Resample the training data with SMOTEENN
        from imblearn.combine import SMOTEENN
        smote_enn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        Counter(y_resampled)

5. Balanced random forest classifier

        # Resample the training data with the BalancedRandomForestClassifier
        from imblearn.ensemble import BalancedRandomForestClassifier
        brfc_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
        brfc_model.fit(X_train, y_train)

6. Ensemble adaboost classifier

        # Train the EasyEnsembleClassifier
        from imblearn.ensemble import EasyEnsembleClassifier
        eec_model = EasyEnsembleClassifier(n_estimators=100,random_state=1)
        eec_model.fit(X_train, y_train)
        
## Summary

Out of the six models, Ensemble adaboost classidier has the highest accuracy score and recall values.  All models have similar precision scores (low performers for high-risk loans).  
