# Credit_Risk_Analysis1

## Overview

This exercise applies and compares six supervised machine learning models:

1. Naive random oversampling (RandomOverSampler library)
2. SMOTE oversampling (SMOTE library) 
3. Undersampling (ClusterCentroids library)
4. Cobmination (over and under) sampling (SMOTEENN algorithm)
5. Random foresSt classifier
6. Easy ensemble adaboost classifier



## Results
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

5. Balanced random forest classifier

      # Resample the training data with the BalancedRandomForestClassifier
      from imblearn.ensemble import BalancedRandomForestClassifier
      brfc_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
      brfc_model.fit(X_train, y_train)
      
5. 
## Summary
