import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# import pandas_profiling

def EDA(a):
    ch_data = a.drop(columns=['math score', 'reading score', 'writing score','percentage','age','IELTS']) #dropping unnecessary columns
    ch_data= ch_data.dropna() 

# Detect outliers using IQR for all numeric columns
    numeric_cols = ch_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = ch_data[col].quantile(0.25)
        Q3 = ch_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((ch_data[col] < lower_bound) | (ch_data[col] > upper_bound)).sum()
        print(f"{col}: {outliers} outliers")
        
        
    for col in numeric_cols:
        Q1 = ch_data[col].quantile(0.25)
        Q3 = ch_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        ch_data = ch_data[(ch_data[col] >= lower_bound) & (ch_data[col] <= upper_bound)]
        
        
        
    print("\n",ch_data.head(),"\n") 
    print("\n",ch_data.info())       #showing the info of the data
    print("\n",ch_data.nunique())
    print("\n",ch_data.isnull().sum()) 
    print("\n",ch_data.describe()) #showing the description of the data
    
    
    temp = input("Do you want to see the Visualization of the data? (y/n): ")
    if(temp == 'y' or temp == 'Y'):    
        for col in ch_data.select_dtypes(include=['object']).columns:
            plt.figure(figsize=(8,4))
            sns.countplot(x=col, data=ch_data)
            plt.title(f'Distribution of {col}')
            plt.show()
        for col in ch_data.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(8,4))
            sns.histplot(ch_data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()
    
    label_encoders = {}
    selected_cols = ['gender','age_group','parental level of education','grade','extracurricular activities','ielts_group','financial sponsorship','visa eligible']
    for col in selected_cols:
        if col in ch_data.columns:
            le = LabelEncoder()
            ch_data[col] = le.fit_transform(ch_data[col])
            label_encoders[col] = le


    plt.figure(figsize=(13,8))
    sns.heatmap(ch_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()


    return ch_data, label_encoders
