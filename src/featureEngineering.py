import numpy as np

def featureEng(df):
    
    # Grade
    df['cred_history_year'] = np.where(df['cb_person_cred_hist_length'] >= 5, '> 5yrs',
                np.where(df['cb_person_cred_hist_length'] < 5, '< 5yrs', '> 10yrs'))  
    

    # Age
    df['age_group'] = np.where(df['person_age'] < 25, 'Young',
                    np.where(df['person_age'] < 40, 'Early Career',
                    np.where(df['person_age'] < 60, 'Stable','Senior')))
    
    return df