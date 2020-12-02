#%%
## ANOMALY DETECTION
#%%

from imblearn.combine import SMOTETomek
import pandas as pd 
import numpy as np 
from numpy import sort, sqrt, argsort, inf
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, precision_recall_curve,confusion_matrix
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# %%
#data = pd.read_csv(r'C:\Users\gcabreram\Google Drive\mi_GitHub\slaG\data\boiler_unity_xxx2.csv',sep=';')
data = pd.read_csv(r'C:\Users\jmruizr\Downloads\boiler_unity_xxx2.csv',sep=';')

selected_colums = ['Active Power','Air Heater #1 Differential Pressure',
    'Boiler Feedwater Pressure', 'Boiler Furnace Pressure', 'Boiler Outlet Pressure',
    'Boosted Overfire Air (Side 1)','Boosted Overfire Air (Side 2)',
    'Coal + Primary Air Temperature @Coal Mill #1 Outlet','Coal + Primary Air Temperature @Coal Mill #2 Outlet',
    'Coal + Primary Air Temperature @Coal Mill #3 Outlet','Coal + Primary Air Temperature @Coal Mill #5 Outlet',
    'Coal Feeder #1',
    'Coal Feeder #2','Coal Feeder #4',
    'Coal Feeder #5','Cold Primary Air Control Damper Position @Coal Mill #1',
    'Cold Primary Air Control Damper Position @Coal Mill #2',
    'Cold Primary Air Control Damper Position @Coal Mill #3',
    'Cold Primary Air Control Damper Position @Coal Mill #4',
    'Cold Primary Air Control Damper Position @Coal Mill #5', 'Dynamic Coal Classifier Rotational Speed Coal Mill #2',
    'Dynamic Coal Classifier Rotational Speed Coal Mill #4', 'Dynamic Coal Classifier Rotational Speed Coal Mill #5',
    'Emergency Reheated Steam Atemperation Valve Position', 'Flue Gas Damper Position 1 (RH Side)',
    'Flue Gas Damper Position 1 (SH Side)', 'Flue Gas Induced Draft Fan #1 Blade Pitch Angle',
    'Flue Gas Induced Draft Fan #1 Current',
    'Flue Gas Induced Draft Fan #1 Flow',
    'Flue Gas Induced Draft Fan #2 Blade Pitch Angle', 'Hot Primary Air Control Damper Position @Coal Mill #1',
    'Hot Primary Air Control Damper Position @Coal Mill #2',
    'Hot Primary Air Control Damper Position @Coal Mill #3', 
    'Main Steam First Desuperheater Control Valve position (Side 1)',
    'Main Steam First Desuperheater Control Valve position (Side 2)',
    'Main Steam Second Desuperheater Control Valve position (Side 1)',
    'Main Steam Second Desuperheater Control Valve position (Side 2)', 'Main Steam Turbine Control Valve #D',
    'Secondary Air Fan #1 Current',  'Secondary Air Fan #2 Current',
    'Secondary Air Row #1 (Side 1)', 'Secondary Air Row #2 (Side 1)', 'Secondary Air Row #2 (Side 2)',
    'Secondary Air Row #3 (Side 1)','Secondary Air Row #3 (Side 2)',
    'Secondary Air Row #4 (Side 1)',
    'Secondary Air Row #4 (Side 2)',
    'Secondary Air Row #5 (Side 1)',
    'Secondary Air Row #5 (Side 2)','SSTIMESTAMP', 'Total Atemperator Feedwater Flow'
    ]

data = data[selected_colums]
na_cols = [x for x in data.columns if x in data.columns[data.isnull().any()].tolist()]
for col in na_cols:
    data[col] = data[col].fillna(data[col].mean())
#%%

data.sort_values(['SSTIMESTAMP'],inplace=True)

#%%
'''
columns = [x for x in data.columns if x not in ['SSTIMESTAMP']]

for col in columns:
    data[col+'_dif'] = data[col]-(data[col].shift(1))
'''
#%%

na_cols = [x for x in data.columns if x in data.columns[data.isnull().any()].tolist()]
for col in na_cols:
    data[col] = data[col].fillna(data[col].mean())
# %%
def q10(x):
    return x.quantile(0.1)
def q20(x):
    return x.quantile(0.2)
def q30(x):
    return x.quantile(0.3)
def q40(x):
    return x.quantile(0.4)
def q50(x):
    return x.quantile(0.5)
def q60(x):
    return x.quantile(0.6)
def q70(x):
    return x.quantile(0.7)
def q80(x):
    return x.quantile(0.8)
def q90(x):
    return x.quantile(0.9)

# %%

poc = data['SSTIMESTAMP']
poc = list(poc)
for i,st in enumerate(poc):
    poc[i] = poc[i][:5]

data['ID'] = poc

#%%
poc = data['SSTIMESTAMP']
daily_poc = list(poc)
hourly_poc = list(poc)
for i,st in enumerate(poc):
    #daily resampling: a partir de -8 el TIME 
    daily_poc[i] = poc[i][-8:]
    #daily resampling: entre -8 y -6 la HORA
    hourly_poc[i] = poc[i][-8:-6]
    
data['TIME'] = daily_poc
data['HOUR'] = hourly_poc

#data.drop(['SSTIMESTAMP'],inplace=True,axis=1)

#%%
poc = data['SSTIMESTAMP']
poc = list(poc)
for i,st in enumerate(poc):
    poc[i] = poc[i][3:5]

data['MES'] = poc

# %%
# TRAINING DATA:
data = data[data['MES'] <= '08']

# %%
data.drop(['SSTIMESTAMP','MES'],inplace=True,axis=1)
data.sort_values(['ID'],inplace=True)

tablon_final = pd.DataFrame()
features = [x for x in data.columns if x not in ['TIME','ID', 'HOUR']]
for col in features:
    prueba = data.groupby(['ID', 'HOUR'])[col].agg(['mean','max','min','std','skew',q10,q20,q30,q40,q50,q60,q70,q80,q90]).reset_index(drop=False)
    prueba = prueba.add_prefix(col+'_')
    tablon_final = pd.concat([tablon_final,prueba],axis=1)
tablon_final['Active Power_ID']= pd.to_datetime(pd.Series(tablon_final['Active Power_ID']), format="%d-%m")
tablon_final.sort_values(['Active Power_ID'],inplace=True)

# %%
tablon_final['TARGET'] = np.where(((tablon_final['Active Power_ID'] <= '1900-08-31') & (tablon_final['Active Power_ID'] >= '1900-08-07')),1,0)
obj_cols = [x for x in tablon_final.select_dtypes(include=['object']).columns]
selected = [ x for x in  tablon_final.columns if x not in obj_cols]
tablon_final = tablon_final[selected]

#%%
tablon_final.drop(['Active Power_ID'], axis=1,inplace=True)

# %%
tablon_final.to_csv('./data/tablon_final.csv')

#%%[markdown]
# NaN values checker:
# %%
def nan_values_checker(dataframe):
    nan_values_percentage = {}
    for attribute in dataframe.columns:
        np_series_nan_values = np.isnan(np.array(dataframe[attribute].values))
        if len(np_series_nan_values[np_series_nan_values==True])==len(dataframe[attribute]):
            dataframe = dataframe.drop(columns=[attribute])
        
        nan_values_percentage[attribute] = len(np_series_nan_values[np_series_nan_values==True])/len(dataframe)
        
    return dataframe, nan_values_percentage


# %%
tablon_no_slag = tablon_final[tablon_final['TARGET']==0]
tablon_slag = tablon_final[tablon_final['TARGET']==1]
#%%
from sklearn import svm

def fit_oc_svm_anomaly_detector(dataframe_attributes, nu_param=0.1, 
                                kernel_param="rbf", gamma_param=0.1 ):
    # fit dataframe
    oc_SVM_anomaly_detector = svm.OneClassSVM(nu=nu_param, kernel=kernel_param, 
                                              gamma=gamma_param)
    oc_SVM_anomaly_detector.fit(dataframe_attributes)
    return oc_SVM_anomaly_detector
#%%
features_columns = [x for x in tablon_no_slag.columns if x not in ['TARGET']]

# %%
def imputeMissingInterpolating(array_df):
    import numpy as np
    for attr in array_df.columns:
        attribute_interpolated = array_df[attr].interpolate(method='linear', limit_direction='both')
        #assert len(attribute_interpolated[np.isnan(attribute_interpolated)]) == 0
        array_df[attr] = attribute_interpolated
    
    return array_df

tablon_no_slag_imputed = imputeMissingInterpolating(tablon_no_slag)
tablon_slag_imputed = imputeMissingInterpolating(tablon_slag)

#%%
features_columns = [x for x in tablon_no_slag.columns if x not in ['TARGET']]

# %%

trained_svc = fit_oc_svm_anomaly_detector(tablon_no_slag[features_columns])
print (tablon_no_slag[features_columns].head())
#%%
        for i in dataframe.index:
            anomaly_score = model.decision_function(np.array(dataframe.iloc[i]).reshape(1, -1))
            scores_list.append(round(anomaly_score[0], 2))
        
        return scores_list
    
    except Exception as exc:
        return exc


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

#%%
# Standard-scaler:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

tablon_no_slag_imputed_fitter = scaler.fit(tablon_no_slag_imputed[selected_attrs].values)
tablon_no_slag_imputed_scaled_values = tablon_no_slag_imputed_fitter.transform(tablon_no_slag_imputed[selected_attrs].values)
tablon_slag_imputed_scaled_values = tablon_no_slag_imputed_fitter.transform(tablon_slag_imputed[selected_attrs].values)

#%%
tablon_no_slag_imputed = pd.DataFrame(columns=selected_attrs, data=tablon_no_slag_imputed_scaled_values)
tablon_slag_imputed = pd.DataFrame(columns=selected_attrs, data=tablon_slag_imputed_scaled_values)

#%%
trained_svc = fit_oc_svm_anomaly_detector(tablon_no_slag_imputed[selected_attrs], nu_param=0.001, 
                                          gamma_param=10)

# %%
tablon_no_slag_imputed.reset_index(inplace=True, drop=True)
target_anomaly_scores = return_scores(trained_svc,tablon_no_slag_imputed[selected_attrs])

# %%
tablon_slag_imputed.reset_index(inplace=True, drop=True)
target_slag_anomaly_scores = return_scores(trained_svc,tablon_slag_imputed[selected_attrs])

#%%
def variable_selector(x_df, y_series, nfolds=3, random_state=42, early_stopping=30):
    '''Selecting important variables based on LightGBM

    Function makes use of boosting feature importance feature. Base estimator
    is LightGBM with non-default hyperparameters. Scheme is simple: iterate
    through variable importances selecting subsets and evaluate loss for each subset.
    Dataset with best cross-validated loss leads to set of best variables. There
    is also early stopping implemented.

    Args:
        x_df: pandas dataframe of predictors.
        y_series: pandas series of target values.
        nfolds: integer, number of cross validation folds.
        random_state: integer, seed for stochastic processes.
        early_stopping: integer, maximum number of iterations without making an
            improvement.

    Returns:
        selected_indices: list of integers containing indexes of best variables.

    '''
    import lightgbm as lgb   

    assert isinstance(y_series, pd.Series),\
    "Please ensure target argument is pandas series type"

    assert isinstance(x_df, pd.DataFrame),\
    "Please ensure dataframe is type pandas DataFrame."

    for _var in (nfolds, random_state, early_stopping):
        assert isinstance(_var, int), "Please ensure {} is type integer".format(_var)

    #setting up parameters for cross validation
    c_v = KFold(nfolds, random_state=random_state, shuffle=True)
    mse_scorer = make_scorer(roc_auc_score)

    #defining constant parameters for LightGBM model
    lgb_params = {
        'bagging_fraction': 0.84588525292245,
        'bagging_freq': 7,
        'colsample_bytree': 0.8968744781143361,
        'feature_fraction': 0.4660296750307845,
        'lambda_l1': 0.0451309695736122,
        'lambda_l2': 1.6927531327628034,
        'learning_rate': 0.010776482097437746,
        'max_bin': 303,
        'max_depth': 23,
        'min_data_in_leaf': 4,
        'min_sum_hessian_in_leaf': 9.773081197656603,
        'n_estimators': 556,
        'num_leaves': 107,
        'subsample': 0.5
        }

    #initializing an instance of the estimator
    clf = lgb.LGBMClassifier(
        n_jobs=-1,
        boosting_type='gbdt',
        objective='binary',
        eval_metric='auc',
        tree_learner='feature',
        silent=False,
        random_state=random_state
        )

    clf.set_params(**lgb_params)
    clf.fit(x_df, y_series)

    #feature importances of variables
    thresholds = sort(list(set(clf.feature_importances_)))

    #initializing variables to hold controlling values
    current_best_roc_auc = -inf
    current_best_variables = x_df.shape[1]
    current_best_index = 0

    for i, thresh in enumerate(thresholds):

        #for every threshold, select from whole dataset variables that match
        #the condition,
        #for such collection of variables, fit and cross validate the estimator
        #and calculate RMSE,
        #if this-very-run stats are better than cached totals, then swap them.

        selection = SelectFromModel(clf, threshold=thresh, prefit=True)
        select_x_train = selection.transform(x_df)

        score = cross_val_score(clf,
                                select_x_train,
                                y_series,
                                scoring=mse_scorer,
                                cv=c_v,
                                verbose=True,
                                n_jobs=-1
                               )

        roc_auc = score.mean()

        if roc_auc > current_best_roc_auc:
            current_best_roc_auc = roc_auc
            current_best_variables = select_x_train.shape[1]
            current_best_index = i
        else:
            continue

        #there is a condition for early stopping implemented, if there is not
        #any updgrade for early_stopping iterations then break the loop

        if i - current_best_index > early_stopping:
            print("Breaking loop due to early stopping condition")
            break
        else:
            continue

        print("Iteration {} out of {}".format(i, len(thresholds)))
        print("Thresh={}, n={}, AUC ROC: {}".format(thresh, select_x_train.shape[1], rmse))

    indices = argsort(clf.feature_importances_)[::-1]
    selected_indices = indices[0:current_best_variables]

    return selected_indices

# %%
tablon_final_imputed = imputeMissingInterpolating(tablon_final)
# %%
inputs = [x for x in tablon_final_imputed.columns if x not in ['TARGET']]
X = tablon_final_imputed[inputs]
y = tablon_final_imputed['TARGET']

smt_sample = SMOTETomek(sampling_strategy ="minority")
X_smt, y_smt = smt_sample.fit_sample(X, y)
y_smt = pd.Series(y_smt)
dataframe_rebalanced = pd.DataFrame(data=X_smt, columns=list(X))

# %%
selected_attrs_indexes = variable_selector(dataframe_rebalanced, y_smt)

# %%
selected_attrs = []
for index in selected_attrs_indexes:
    selected_attrs.append(features_columns[index])

# %%

target = return_scores(trained_svc,tablon_no_slag[features_columns])

# %%

target_slag = return_scores(trained_svc,tablon_slag[features_columns])


# %%
