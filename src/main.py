from EDA import EDA, encode
from naiveBayes import nb_model
from decTree import tree_model 
from rForest import rfc_model 
from GUI import launch_gui
from evalPrint import evalP
from featureEngineering import featureEng 
import pandas as pd
import numpy as np


def main():
    data= pd.read_csv('src/visa_eligible.csv')
    n_data=featureEng(data)  
    e_data=EDA(n_data)
    ch_data,lb=encode(e_data.copy())   
    yt,yp,x,le,md=rfc_model(ch_data,lb)
    # evalP(yt,yp,x,le,rf)
    accuracy = (yt == yp).mean() * 100
    launch_gui(accuracy, yt, yp, x, le, md, ch_data, e_data)

if __name__ == "__main__":
    main() 