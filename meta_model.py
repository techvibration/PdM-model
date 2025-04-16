import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 

# Load your saved models
model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')

# Loading the dataset
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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify = y,random_state=42)

# Define the base models for stacking
estimators = [
    ('xgb1', model1),
    ('xgb2', model2)
]

# Choose a meta-learner 
meta_learner = LogisticRegression()

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_learner, cv=5)

# Train the stacked model
stacking_model.fit(X_train, y_train)

# Make predictions on the test data
final_predictions = stacking_model.predict(X_test)

# Evaluate the stacked model
print(classification_report(y_test, final_predictions))
print("Confusion Matrix:")
print (confusion_matrix(y_test,final_predictions))
#Calculate the roc-aur score
roc_auc = roc_auc_score(y_test,final_predictions)
print("ROC-AUC score:",roc_auc)
joblib.dump(stacking_model,'model3.pkl')
print("model saved as model3.pkl")
