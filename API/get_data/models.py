from tkinter import Y
from django.db import models
from django.http import JsonResponse
import json
# Create your models here.

class Clusters(models.Model):
    
    @staticmethod
    def getClusters(body_json):
        import matplotlib
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import csv
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples, silhouette_score
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import MinMaxScaler
        from sklearn import metrics
        from sklearn.feature_selection import SelectKBest
        import pickle
        from sklearn import metrics
        from sklearn.feature_selection import SelectKBest   #for feature selection
        from sklearn.feature_selection import f_classif
        
        df_train = pd.read_csv('R:\REST API\API\get_data\TRAIN.csv')
        df_test = pd.read_csv('R:\REST API\API\get_data\TEST.csv')
        
        categorical = df_train.select_dtypes(include =object)
        numerical= df_train.select_dtypes(include =[np.float64,np.int64])

        s = ['examide', 'citoglipton', 'glimepiride-pioglitazone', 'metformin-pioglitazone', 'metformin-rosiglitazone']
        df_train.drop(columns=s, inplace=True)
    
        df_test.drop(columns=s, inplace=True)

        df_test.replace('?', np.nan, inplace = True)
        df_train.replace('?', np.nan, inplace = True)
        
        
        #lets find the column containing the nan values for train and test set
        train_na_names = df_train.isnull().any()

        test_na_names = df_test.isnull().any()

        # lets fill the nan value for train dataset
        df_train['race'] = df_train['race'].fillna(df_train['race'].mode()[0])
        df_train['weight'] = df_train['weight'].fillna(df_train['weight'].mode()[0])
        df_train['payer_code'] = df_train['payer_code'].fillna(df_train['payer_code'].mode()[0])
        df_train['medical_specialty'] = df_train['medical_specialty'].fillna(df_train['medical_specialty'].mode()[0])
        df_train['diag_1'] = df_train['diag_1'].fillna(df_train['diag_1'].mode()[0])
        df_train['diag_2'] = df_train['diag_2'].fillna(df_train['diag_2'].mode()[0])
        df_train['diag_3'] = df_train['diag_3'].fillna(df_train['diag_3'].mode()[0])

        # lets fill the nan value for test dataset
        df_test['race'] = df_test['race'].fillna(df_test['race'].mode()[0])
        df_test['weight'] = df_test['weight'].fillna(df_test['weight'].mode()[0])
        df_test['payer_code'] = df_test['payer_code'].fillna(df_test['payer_code'].mode()[0])
        df_test['medical_specialty'] = df_test['medical_specialty'].fillna(df_test['medical_specialty'].mode()[0])
        df_test['diag_1'] = df_test['diag_1'].fillna(df_test['diag_1'].mode()[0])
        df_test['diag_2'] = df_test['diag_2'].fillna(df_test['diag_2'].mode()[0])
        df_test['diag_3'] = df_test['diag_3'].fillna(df_test['diag_3'].mode()[0])

        df_train.drop_duplicates(inplace=True)

        input_cols = df_train.columns
        numeric_cols = df_train[input_cols].select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_train[input_cols].select_dtypes(include='object').columns.tolist()

        # encoding the categorical column in train_df
        from sklearn.preprocessing import LabelEncoder
        LB =LabelEncoder()
        for i in categorical_cols:
            LB.fit(df_train[i])
            df_train[i] = LB.transform(df_train[i])
            

        # encoding the categorical column in test_df
        from sklearn.preprocessing import LabelEncoder
        LB =LabelEncoder()
        for i in categorical_cols:
            LB.fit(df_test[i])
            df_test[i] = LB.transform(df_test[i])
            

            
        # Impute and scale numeric columns for train_df



        imputer = SimpleImputer().fit(df_train[numeric_cols])
        df_train[numeric_cols] = imputer.transform(df_train[numeric_cols])
        scaler = MinMaxScaler().fit(df_train[numeric_cols])
        df_train[numeric_cols] = scaler.transform(df_train[numeric_cols])


        # Impute and scale numeric columns for test_df



        imputer = SimpleImputer().fit(df_test[numeric_cols[:-1]])
        df_test[numeric_cols[:-1]] = imputer.transform(df_test[numeric_cols[:-1]])
        scaler = MinMaxScaler().fit(df_test[numeric_cols[:-1]])
        df_test[numeric_cols[:-1]] = scaler.transform(df_test[numeric_cols[:-1]])

        test = SelectKBest(score_func=f_classif, k=25)
        fit = test.fit(df_train.drop(columns=['readmitted_NO']), df_train['readmitted_NO'])
        a = [i for i in fit.feature_names_in_ if i not in fit.get_feature_names_out()]
        len(a)

        index_test = df_test['index']
        df_test.drop(columns='index', inplace=True)

        target1 = df_train['readmitted_NO']
        df_train.drop(columns='readmitted_NO', inplace=True)
        df_train.drop(columns= a, inplace=True)
        best_cols = fit.get_feature_names_out()
        # Load from file
        with open('./get_data/pickle_model.pkl', 'rb') as file:
            pickle_model = pickle.load(file)
            
        Ypredict = pickle_model.predict(df_test[best_cols])
        lists = Ypredict.tolist()
        json_str = json.dumps(lists)
       
                
        
        return JsonResponse({"Cluster_no":json_str})