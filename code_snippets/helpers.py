#%%[markdown]
# NaN values checker:
# %%
def nan_values_checker(dataframe):
    import numpy as np 

    nan_values_percentage = {}
    for attribute in dataframe.columns:
        np_series_nan_values = np.isnan(np.array(dataframe[attribute].values))
        if len(np_series_nan_values[np_series_nan_values==True])==len(dataframe[attribute]):
            dataframe = dataframe.drop(columns=[attribute])
        
        nan_values_percentage[attribute] = len(np_series_nan_values[np_series_nan_values==True])/len(dataframe)
        
    return dataframe, nan_values_percentage

# %%
def imputeMissingInterpolating(array_df):
    import numpy as np

    for attr in array_df.columns:
        attribute_interpolated = array_df[attr].interpolate(method='linear', limit_direction='both')
        #assert len(attribute_interpolated[np.isnan(attribute_interpolated)]) == 0
        
        array_df[attr] = attribute_interpolated
    
    return array_df

#%%[markdown]
# ANomaly detector with OC-SVM
def fit_oc_svm_anomaly_detector(dataframe_attributes, nu_param=0.01, 
                                kernel_param="rbf", gamma_param=0.01 ):
    
    from sklearn.svm import OneClassSVM
    # fit dataframe
    oc_SVM_anomaly_detector = OneClassSVM(nu=nu_param, kernel=kernel_param, 
                                              gamma=gamma_param)
    oc_SVM_anomaly_detector.fit(dataframe_attributes)

    return oc_SVM_anomaly_detector

def return_scores(model, dataframe):
    import numpy as np
    """
      - receives: 
        * trained anomaly detector
        * dataframe: attributes on which the detector will return the scores 
    """
    scores_list = []
    try:
        for i in dataframe.index:
            anomaly_score = model.decision_function(np.array(dataframe.iloc[i]).reshape(1, -1))
            scores_list.append(round(anomaly_score[0], 2))
        
        return scores_list
    
    except Exception as exc:
        return exc

# %%
