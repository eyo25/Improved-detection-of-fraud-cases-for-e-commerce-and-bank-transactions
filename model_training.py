from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input

# Load preprocessed data
fraud_data = pd.read_csv('../data/preprocessed_fraud_data.csv')
creditcard_data = pd.read_csv('../data/creditcard.csv')

# Feature and Target Separation
X_fraud = fraud_data.drop(columns=['class'])
y_fraud = fraud_data['class']
X_creditcard = creditcard_data.drop(columns=['Class'])
y_creditcard = creditcard_data['Class']

# Validate that there are no non-numeric values
X_fraud = X_fraud.apply(pd.to_numeric, errors='coerce')
X_creditcard = X_creditcard.apply(pd.to_numeric, errors='coerce')
y_fraud = y_fraud.apply(pd.to_numeric, errors='coerce')
y_creditcard = y_creditcard.apply(pd.to_numeric, errors='coerce')

# Ensure no missing values in the datasets after conversion
X_fraud.fillna(0, inplace=True)
X_creditcard.fillna(0, inplace=True)
y_fraud.fillna(0, inplace=True)
y_creditcard.fillna(0, inplace=True)


# Train-Test Split
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = train_test_split(X_creditcard, y_creditcard, test_size=0.2, random_state=42)

# Function to log models and results using MLflow
def log_model_results(model_name, model, X_test, y_test, is_neural_net=False):
    with mlflow.start_run():
        if is_neural_net:
            predictions = (model.predict(X_test) > 0.5).astype("int32")
        else:
            predictions = model.predict(X_test)
        mlflow.sklearn.log_model(model, model_name)
        auc_score = roc_auc_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("roc_auc", auc_score)
        mlflow.log_metric("accuracy", accuracy)
        print(f"{model_name} ROC AUC: {auc_score}")
        print(classification_report(y_test, predictions))

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_fraud, y_train_fraud)
log_model_results("Logistic Regression - Fraud Data", lr, X_test_fraud, y_test_fraud)

lr.fit(X_train_creditcard, y_train_creditcard)
log_model_results("Logistic Regression - Credit Card Data", lr, X_test_creditcard, y_test_creditcard)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_fraud, y_train_fraud)
log_model_results("Decision Tree - Fraud Data", dt, X_test_fraud, y_test_fraud)

dt.fit(X_train_creditcard, y_train_creditcard)
log_model_results("Decision Tree - Credit Card Data", dt, X_test_creditcard, y_test_creditcard)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_fraud, y_train_fraud)
log_model_results("Random Forest - Fraud Data", rf, X_test_fraud, y_test_fraud)

rf.fit(X_train_creditcard, y_train_creditcard)
log_model_results("Random Forest - Credit Card Data", rf, X_test_creditcard, y_test_creditcard)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_fraud, y_train_fraud)
log_model_results("Gradient Boosting - Fraud Data", gb, X_test_fraud, y_test_fraud)

gb.fit(X_train_creditcard, y_train_creditcard)
log_model_results("Gradient Boosting - Credit Card Data", gb, X_test_creditcard, y_test_creditcard)

# Multi-Layer Perceptron (MLP)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train_fraud, y_train_fraud)
log_model_results("MLP - Fraud Data", mlp, X_test_fraud, y_test_fraud)

mlp.fit(X_train_creditcard, y_train_creditcard)
log_model_results("MLP - Credit Card Data", mlp, X_test_creditcard, y_test_creditcard)

# Convolutional Neural Network (CNN)
# Convolutional Neural Network (CNN)
# Convolutional Neural Network (CNN)
def create_cnn(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Expand dimensions for CNN
X_train_fraud_cnn = np.expand_dims(X_train_fraud, axis=-1)
X_test_fraud_cnn = np.expand_dims(X_test_fraud, axis=-1)
print(f'X_train_fraud_cnn shape: {X_train_fraud_cnn.shape}')
print(f'X_test_fraud_cnn shape: {X_test_fraud_cnn.shape}')

cnn = create_cnn((X_train_fraud_cnn.shape[1], 1))
cnn.fit(X_train_fraud_cnn, y_train_fraud, epochs=10, batch_size=32, validation_split=0.2)
log_model_results("CNN - Fraud Data", cnn, X_test_fraud_cnn, y_test_fraud, is_neural_net=True)

X_train_creditcard_cnn = np.expand_dims(X_train_creditcard, axis=-1)
X_test_creditcard_cnn = np.expand_dims(X_test_creditcard, axis=-1)
print(f'X_train_creditcard_cnn shape: {X_train_creditcard_cnn.shape}')
print(f'X_test_creditcard_cnn shape: {X_test_creditcard_cnn.shape}')

cnn_creditcard = create_cnn((X_train_creditcard_cnn.shape[1], 1))
cnn_creditcard.fit(X_train_creditcard_cnn, y_train_creditcard, epochs=10, batch_size=32, validation_split=0.2)
log_model_results("CNN - Credit Card Data", cnn_creditcard, X_test_creditcard_cnn, y_test_creditcard, is_neural_net=True)


# Recurrent Neural Network (RNN) / Long Short-Term Memory (LSTM)
def create_lstm(input_shape):
    model = Sequential([
        LSTM(100, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train_fraud_lstm = np.expand_dims(pad_sequences(X_train_fraud.values), axis=-1)
X_test_fraud_lstm = np.expand_dims(pad_sequences(X_test_fraud.values), axis=-1)

print(f'X_train_fraud_lstm shape: {X_train_fraud_lstm.shape}')
print(f'X_test_fraud_lstm shape: {X_test_fraud_lstm.shape}')

lstm_fraud = create_lstm((X_train_fraud_lstm.shape[1], 1))
lstm_fraud.fit(X_train_fraud_lstm, y_train_fraud, epochs=10, batch_size=32, validation_split=0.2)
log_model_results("LSTM - Fraud Data", lstm_fraud, X_test_fraud_lstm, y_test_fraud, is_neural_net=True)

# Preprocessing and training for credit card data
X_train_creditcard_lstm = np.expand_dims(pad_sequences(X_train_creditcard.values), axis=-1)
X_test_creditcard_lstm = np.expand_dims(pad_sequences(X_test_creditcard.values), axis=-1)

print(f'X_train_creditcard_lstm shape: {X_train_creditcard_lstm.shape}')
print(f'X_test_creditcard_lstm shape: {X_test_creditcard_lstm.shape}')

lstm_creditcard = create_lstm((X_train_creditcard_lstm.shape[1], 1))
lstm_creditcard.fit(X_train_creditcard_lstm, y_train_creditcard, epochs=10, batch_size=32, validation_split=0.2)
log_model_results("LSTM - Credit Card Data", lstm_creditcard, X_test_creditcard_lstm, y_test_creditcard, is_neural_net=True)
