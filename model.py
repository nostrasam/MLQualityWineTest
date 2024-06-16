import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define paths to data files
train_data_path = 'wine_train.csv'
test_data_path = 'wine_test.csv'

# Load the training data
wtrain = pd.read_csv(train_data_path)

# Preprocess the training data
wtrain['type'] = wtrain['type'].map({'white': 1, 'red': 0})
X_train = wtrain.drop('quality', axis=1)
y_train = wtrain['quality']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model, scaler, and imputer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('imputer.pkl', 'wb') as imputer_file:
    pickle.dump(imputer, imputer_file)

# Load the test data
wtest = pd.read_csv(test_data_path)

# Preprocess the test data
wtest['type'] = wtest['type'].map({'white': 1, 'red': 0})
X_test = wtest.drop('quality', axis=1)
y_test = wtest['quality']

# Handle missing values by imputing with the mean
X_test = imputer.transform(X_test)

# Scale the features
X_test = scaler.transform(X_test)

# Making prediction using the loaded model on the test dataset
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error on test set: {mse}')
print(f'R^2 Score on test set: {r2}')
