from tkinter import Y
from django.db import models
from django.http import JsonResponse
import json
# Create your models here.

class Clusters(models.Model):
    
    @staticmethod
    def getClusters(df_test0):
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
        import hdfs3
        print("here2")
        
        df_test = pd.DataFrame(df_test0)
        print("here3")
        print(df_test.columns)
        categorical = df_test.select_dtypes(include =object)
        numerical= df_test.select_dtypes(include =[np.float64,np.int64])

        
        df_test.replace('?', np.nan, inplace = True)
        
        
        
        #lets find the column containing the nan values for train and test set
       

        test_na_names = df_test.isnull().any()

        

        # lets fill the nan value for test dataset
     
        df_test['medical_specialty'] = df_test['medical_specialty'].fillna('3')
                     

        df_test.drop_duplicates(inplace=True)

        input_cols = df_test.columns
        numeric_cols = df_test[input_cols].select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_test[input_cols].select_dtypes(include='object').columns.tolist()

        
            

        # encoding the categorical column in test_df
        from sklearn.preprocessing import LabelEncoder
        LB =LabelEncoder()
        for i in categorical_cols:
            LB.fit(df_test[i])
            df_test[i] = LB.transform(df_test[i])
            

            
       

        # Impute and scale numeric columns for test_df



        imputer = SimpleImputer().fit(df_test[numeric_cols[:-1]])
        df_test[numeric_cols[:-1]] = imputer.transform(df_test[numeric_cols[:-1]])
        scaler = MinMaxScaler().fit(df_test[numeric_cols[:-1]])
        df_test[numeric_cols[:-1]] = scaler.transform(df_test[numeric_cols[:-1]])

        
        

        best_cols = ['time_in_hospital', 'medical_specialty', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'change', 'diabetesMed']
        # Load from file
        with open('./get_data/pickle_model.pkl', 'rb') as file:
            pickle_model = pickle.load(file)
            
        Ypredict = pickle_model.predict(df_test[best_cols])
        lists = Ypredict.tolist()
        json_str = json.dumps(lists)
        print('done with the API')
                
        
        return JsonResponse({"Cluster_no":json_str})