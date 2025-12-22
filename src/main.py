from EDA import EDA 
from naiveBayes import nb_model
from decTree import tree_model 
from featureEngineering import featureEng 
import pandas as pd
import numpy as np


def main():
    data= pd.read_csv('src/visa_eligible.csv')
    n_data=featureEng(data)  
    ch_data,lb=EDA(n_data)
    tree_model(ch_data,lb)

if __name__ == "__main__":
    main() 