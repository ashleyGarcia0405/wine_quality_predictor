import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('winequality.csv')

# first five rows + info
print(df.head())
df.info()
print(df.describe().T)

# input missing values with the mean of respective columns
print(df.isnull().sum())
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

print(df.isnull().sum().sum())

# visualize with histograms
df.hist(bins=20, figsize=(10, 10))
plt.show()

# count plot for each quality of wine
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

# correlation heatmap for the features
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

# drop 'total sulfur dioxide'
df = df.drop('total sulfur dioxide', axis=1)

# Create a binary classification target variable 'best quality'
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

# Replace 'white' with 1 and 'red' with 0 in the dataset (if applicable)
df.replace({'white': 1, 'red': 0}, inplace=True)

# Separate features and target variable
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

# Split the data into training and testing sets (80:20)
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
print(xtrain.shape, xtest.shape)

# Normalize the data
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Initialize models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

# Train and evaluate each model
for model in models:
    model.fit(xtrain, ytrain)
    train_acc = metrics.roc_auc_score(ytrain, model.predict(xtrain))
    val_acc = metrics.roc_auc_score(ytest, model.predict(xtest))
    print(f'{model} : ')
    print('Training Accuracy : ', train_acc)
    print('Validation Accuracy : ', val_acc)
    print()

# Evaluate the best model (example: Logistic Regression)
best_model = models[0]  # Assume Logistic Regression is the best based on above results
metrics.plot_confusion_matrix(best_model, xtest, ytest)
plt.show()

# Print classification report for the best model
print(metrics.classification_report(ytest, best_model.predict(xtest)))

