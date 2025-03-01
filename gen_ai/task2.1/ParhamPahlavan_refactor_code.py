import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load dataset
DATA_PATH = 'weatherAUS.csv'
data = pd.read_csv(DATA_PATH)

# Drop irrelevant or redundant features
DROP_COLUMNS = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
data.drop(columns=DROP_COLUMNS, inplace=True)

# Handle missing values
data.replace('NA', np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and target variable
y = data['RainTomorrow']
X = data.drop(columns='RainTomorrow')

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# Initialize and train MLP Classifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(40, 40, 40),
    activation='logistic',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='adaptive',
    shuffle=True,
    max_iter=1000,
    random_state=100
)
mlp_model.fit(X_train, y_train)

# Evaluate model
accuracy = mlp_model.score(X_test, y_test)
print(f"MLP Model's accuracy: {accuracy * 100:.2f}%")
