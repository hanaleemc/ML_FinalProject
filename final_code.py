### Part 0: Set up Environment 

# Load required librairies 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import tensorflow as tf
import keras
from keras import layers, regularizers
from keras import activations

# Load data with pandas 
fire_data = pd.read_csv('fires.csv') 

## 0.1 Data Exploration 
'''
#check for missing data 
fire_data.isnull().sum()

# check distribution of area column based on project description 
plt.figure(figsize=(8, 6))
plt.hist(fire_data['area'], bins=50, color='blue', edgecolor='black')
plt.title('Area Distribution')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.show()'''

### Part 1: Data Preprocessing 

# 1.1 Implement Log transformation to Area column 
fire_data['log_area'] = np.log(fire_data['area'] + 1)  

# check distribution of log area column post-transformation  
'''plt.figure(figsize=(8, 6))
plt.hist(fire_data['log_area'], bins=50, color='blue', edgecolor='black')
plt.title('Log Area Distribution')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.show()'''

fire_data = fire_data.drop('area',axis=1)

# 1.2 Implement One-Hot Encoding on categorical columns 
fire_data = pd.get_dummies(fire_data, columns=['month', 'day'], drop_first=True)

#### Part 2 Model Selection & Training¶

# 2.1 Preparing data for model 

# Split into features and target variables
X = fire_data.drop('log_area', axis=1) #feature variables
y = fire_data['log_area'] #target variable 

# Split the data into train (70%) and temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=20)

# Split the temporary set into validation (15%) and test sets (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# Scale features using scaling method introduced in class 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 2.2  Implement Baseline Models

# Implement Mean Predictor
mean_pred = np.full_like(y_val, y_train.mean()) 
mean_mse = mean_squared_error(y_val, mean_pred)

# Median Predictor
median_pred = np.full_like(y_val, np.median(y_train))
median_mse = mean_squared_error(y_val, median_pred)

# 2.3 Implement Random Forest Model
RF_model = RandomForestRegressor(n_estimators=100, random_state=20)
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_val)
mse_RF = mean_squared_error(y_val, y_pred_RF)

# 2.4. Implement Support Vector Machine | RBF Kernel
svm_model1 = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svm_model1.fit(X_train, y_train)
y_pred_svm1 = svm_model1.predict(X_val)
mse_svm1 = mean_squared_error(y_val, y_pred_svm1)

# 2.5 Implment Neural Network Model 

# Function to build a Neural Network with Leaky ReLU activation function, elastic net regularization, he initialization
def build_nn_model(input_shape, hidden_layers,l1=0.01, l2=0.01):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    
    # Leaky ReLU as activation function in hidden layers 
    for units in hidden_layers:
        model.add(layers.Dense(units, 
                               activation=activations.get('leaky_relu'), 
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),  
                               kernel_initializer='he_normal'))
    model.add(layers.Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Implement NN with 1 hidden layer
nn_model1 = build_nn_model(X_train.shape[1], [100]) 

# Compile the model
nn_model1.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
nn_model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)
y_pred_nn1 = nn_model1.predict(X_val)
mse_nn1 = mean_squared_error(y_val, y_pred_nn1)


# Implement NN with 2 hidden layer
nn_model2 = build_nn_model(X_train.shape[1], [100, 50]) 
# Compile the model
nn_model2.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
nn_model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)
y_pred_nn2 = nn_model2.predict(X_val)
mse_nn2 = mean_squared_error(y_val, y_pred_nn2)


# 2.6 Model Comparison Table
model_results = {
    'Model': ['Baseline-Mean', 'Baseline-Median', 'Random Forest', 'SVM [RBF]', 'Neural Network(1 Layer)', 'Neural Network (2 Layers)'],
    'MSE': [mean_mse, median_mse, mse_RF, mse_svm1, mse_nn1, mse_nn2]
}

model_comparison1 = pd.DataFrame(model_results)
#print(model_comparison1)

# 2.7 Test Best Model on Test Set 
best_model = nn_model2  
y_pred_test = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Test MSE of the best model (NN 2 layer) on full feature set: {test_mse}")


### Part 3 : Feature Reduction 

# 3.1 Reduce features & prep for model training 

# Split features and target
X = fire_data[['temp','RH','wind','rain']]
y = fire_data['log_area'] #target variable 

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Split the data into train (70%) and temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=20)

# Split the temporary set into validation (15%) and test set (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# Scale features using scaling method introduced in class 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 3.2 Re-implement Baseline Models

# Implement Mean Predictor
mean_pred = np.full_like(y_val, y_train.mean()) 
mean_mse2 = mean_squared_error(y_val, mean_pred)

# Median Predictor
median_pred = np.full_like(y_val, np.median(y_train))
median_mse2 = mean_squared_error(y_val, median_pred)

# 3.3 Re-train Random Forest Model
RF_model = RandomForestRegressor(n_estimators=100, random_state=20)
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_val)
mse_RF2 = mean_squared_error(y_val, y_pred_RF)

# 3.4 Re-train Support Vector Machine | RBF Kernel
svm_model2 = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svm_model2.fit(X_train, y_train)
y_pred_svm2 = svm_model2.predict(X_val)
mse_svm2 = mean_squared_error(y_val, y_pred_svm2)

# 3.5 Re-train Neural Network 

# NN with 1 hidden layer
nn_model1 = build_nn_model(X_train.shape[1], [100]) 
nn_model1.compile(optimizer='adam', loss='mean_squared_error')
nn_model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)
y_pred_nn1 = nn_model1.predict(X_val)
mse_nn1_2 = mean_squared_error(y_val, y_pred_nn1)


#  NN with 2 hidden layer
nn_model2 = build_nn_model(X_train.shape[1], [100, 50]) 
nn_model2.compile(optimizer='adam', loss='mean_squared_error')
nn_model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)
y_pred_nn2 = nn_model2.predict(X_val)
mse_nn2_2 = mean_squared_error(y_val, y_pred_nn2)

# 3.6 Model Comparison Table
model_results2 = {
    'Model': ['Baseline-Mean', 'Baseline-Median', 'Random Forest', 'SVM [RBF]', 'Neural Network(1 Layer)', 'Neural Network (2 Layers)'],
    'MSE_1': [mean_mse, median_mse, mse_RF, mse_svm1, mse_nn1, mse_nn2],
    'MSE_2': [mean_mse2, median_mse2, mse_RF2, mse_svm2, mse_nn1_2, mse_nn2_2]
}

model_comparison2 = pd.DataFrame(model_results2)
print(model_comparison2)

# 3.7 Test Best Model on Test Set 
best_model = nn_model2  
y_pred_test = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Test MSE of the best model (Neural Network 2) on reduced feature set: {test_mse}")
