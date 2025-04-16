import numpy as np
import pandas as pd
import xgboost as xgb
#print("XGBoost version:", xgb.__version__)
from imblearn.combine import SMOTETomek
import optuna   
import joblib
from xgboost import XGBClassifier
from xgboost import callback
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import cross_val_score,cross_val_predict,RepeatedStratifiedKFold 
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

######

dataset_path = "D:\\PdM_model\\data\\projectdataset\\ai4i2020.csv"
pdm_dataset = pd.read_csv(dataset_path)
print(pdm_dataset['Product ID'].nunique())
pdm_dataset.drop(['UDI','Product ID'],axis = 1,inplace = True)
#print(pdm_dataset.dtypes)
ct = ColumnTransformer(
    transformers = [
        ('model_type',OneHotEncoder(),['Type'])
    ],
    remainder ='passthrough' #All other columns will be left unchanged
)
#Fit and transform the data
transformed_data = ct.fit_transform(pdm_dataset)
ohe_type = ct.named_transformers_['model_type'].get_feature_names_out(['Type'])
new_dataset_columns = list(ohe_type) + ['Air_temp'] + ['Process_temp'] + ['Rotational_speed'] + ['Torque'] + ['Tool_wear'] + ['Machine_failure'] + ['TWF'] + ['HDF'] + ['PWF'] + ['OSF'] +['RNF']
pdm_dataset_transformed = pd.DataFrame(transformed_data, columns = new_dataset_columns)
#print(pdm_dataset_transformed.head())
#print(list(ohe_type))
pdm_dataset_transformed.drop(['TWF','HDF','PWF','OSF','RNF'],axis = 1,inplace = True)
y = pdm_dataset_transformed['Machine_failure']
X = pdm_dataset_transformed.drop('Machine_failure',axis = 1)
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2,stratify = y,random_state = 42) 
test_X.to_csv('X_val.csv', index=False)
test_y.reset_index(drop=True).to_frame(name='Machine failure').to_csv('y_val.csv', index=False)

#using scale_pos_weight to increase recall
scale_pos_weight = (train_y == 0).sum() / (train_y == 1).sum()
adjusted_scale_pos_weight = scale_pos_weight * 2.5
#Instantiating smotetomek for balancing the data
#smotetomek = SMOTETomek(sampling_strategy = 0.5,random_state=42)
#Resample the training data
#train_X_rsmpl,train_y_rsmpl = smotetomek.fit_resample(train_X,train_y)
#Define Your XGBoost model
#setting up repeated stratified k-fold cross-validation
cv = RepeatedStratifiedKFold(n_splits = 5,n_repeats = 10,random_state = 42)
#defining the objective function for optuna 
#def objective(trial):
# params = {
#         'n_estimators' : trial.suggest_int('n_estimators',100,1000),
#            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#'max_depth': trial.suggest_int('max_depth', 3, 10),
# 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#  'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#   'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#    'scale_pos_weight': adjusted_scale_pos_weight,
#     'use_label_encoder': False,
#      'eval_metric': 'logloss'
#}
     
    
#pdm_model = XGBClassifier(**params)
#print(pdm_model.fit)
#pdm_model.fit(train_X,train_y,eval_set =[(test_X,test_y)],verbose = True) 

    #performing cross-Validation
#  CV_scores = cross_val_score(pdm_model,train_X,train_y,cv = cv,scoring = 'roc_auc').mean()
#  return CV_scores
#Creating a study to optimize the objective function
#study = optuna.create_study(direction ='maximize')
#study.optimize(objective,n_trials = 50)
# Output the best hyperparameters found by Optuna
#print("Best trial:")
#trial = study.best_trial
#print("  ROC-AUC: {}".format(trial.value))
#print("  Best hyperparameters: {}".format(trial.params))
best_params = {'n_estimators': 797, 'learning_rate': 0.013447616611294044, 'max_depth': 5, 'min_child_weight': 9, 'subsample': 0.777144970227899, 'colsample_bytree': 0.9552404535429462}
pdm_model = XGBClassifier(**best_params,use_label_encoder = False,eval_metric = 'logloss',scale_pos_weight = adjusted_scale_pos_weight)
pdm_model.fit(train_X,train_y,eval_set =[(test_X,test_y)],verbose = True)

CV_scores = cross_val_score(pdm_model,train_X,train_y,cv = cv,scoring = 'roc_auc')
print("Cross Validation score",CV_scores)
print("Mean Cross Validation score",np.mean(CV_scores))

predictions = pdm_model.predict(test_X)
probablity = pdm_model.predict_proba(test_X)[:,1]
#setting threshold to increase recall of the failure class
threshold = 0.4
custom_predictions = (probablity >= threshold).astype(int)
#printing the classification report for precision,recall,f1score
print("Classification Report:")
print(classification_report(test_y,custom_predictions))
#printing the confusion matrix
print("Confusion Matrix:")
print (confusion_matrix(test_y,custom_predictions))
#Calculate the roc-aur score
roc_auc = roc_auc_score(test_y,custom_predictions)
print("ROC-AUC score:",roc_auc)
joblib.dump(pdm_model,'model1.pkl')
print("model saved as model1.pkl")


# Save the ColumnTransformer
joblib.dump(ct, 'ct.pkl')
print("Preprocessor saved as ct.pkl")





