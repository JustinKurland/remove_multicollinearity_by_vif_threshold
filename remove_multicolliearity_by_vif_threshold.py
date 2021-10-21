import pandas as pd
import numpy as np
import pandas_flavor as pf
from statsmodels.stats.outliers_influence import variance_inflation_factor    

@pf.register_dataframe_method
def remove_multicollinearity_by_vif_threshold(df, vif_threshold = 10.0):
    """
    Uses the variance inflation factors between features and a 
    specified threshold to identify and remove collinear/multicollinear features.

    Args:
        df ([pandas.DataFrame]): 
            A dataframe that includes all the features that are being considered 
            for modeling without the target.df ([type]): [description]
        vif_threshold (float, optional): 
            Defaults to 10.0 as per the seminal work of Marquardt DW 
            (1970) Generalized inverses, ridge regression, biased linear 
            estimation and nonlinear estimation. Technometrics 12(3):591–612. Other
            rules of thumb around VIF suggest a value of as low as 4. 

    Returns:
        [pandas.DataFrame]:
            A Pandas Dataframe that has removed all collinear/multicollinear features from 
            the feature space based upon the correlation threshold. 
    """
    variables = list(range(df.shape[1]))
    dropped   = True

    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df.iloc[:, variables].values, i)
               for i in range(df.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > vif_threshold:
            print('dropping \'' + df.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(df.columns[variables])
    
    return df.iloc[:, variables]
