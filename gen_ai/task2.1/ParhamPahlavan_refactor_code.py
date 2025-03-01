import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load dataset
data = pd.read_csv('weatherAUS.csv')

# Drop unnecessary columns
drop_columns = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
data.drop(columns=drop_columns, inplace=True)

# Handle missing values
data.replace('NA', np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and target variable
X = data.drop(columns=['RainTomorrow'])
y = data['RainTomorrow']

# Normalize the feature data
X = StandardScaler().fit_transform(X)

# Split data into training and test sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.25, random_state=100)

# Define and train the MLP classifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(40, 40, 40),
    activation='logistic',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=100
)

mlp_model.fit(trainX, trainY)

# Evaluate model performance
accuracy = mlp_model.score(testX, testY)
print(f"MLP Model's accuracy: {accuracy * 100:.2f}%")
