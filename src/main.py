from EDA import EDA, encode
from naiveBayes import nb_model
from decTree import tree_model 
from rForest import rfc_model 
from GUI import launch_gui
from evalPrint import evalP
from featureEngineering import featureEng
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    data= pd.read_csv('src/credit_risk.csv')
    n_data=featureEng(data)  
    e_data=EDA(n_data)
    ch_data,lb=encode(e_data.copy())
    

    X = ch_data.drop('loan_status', axis=1)
    y = ch_data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    

    yt_nb, yp_nb, ys_nb, x, le, md, nacc = nb_model(X_train, X_test, y_train, y_test, lb)
    yt_dt, yp_dt, ys_dt, x, le, md, dacc = tree_model(X_train, X_test, y_train, y_test, lb)
    yt_rf, yp_rf, ys_rf, x, le, md, racc = rfc_model(X_train, X_test, y_train, y_test, lb)

    # evalP(yt_nb,yp_nb,x,le,md)
    launch_gui((racc*100), yt_rf, yp_rf, x, le, md,y_test, ys_nb, ys_dt, ys_rf ,ch_data, e_data)

if __name__ == "__main__":
    main() 
    
    

