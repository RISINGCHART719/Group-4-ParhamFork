import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('housing_price_dataset.csv')

# Encode categorical variable 'Neighborhood' into numerical values
le = LabelEncoder()
data["Neighborhood"] = le.fit_transform(data["Neighborhood"])

# Separate target variable (Price) and features
y_values = data['Price']
x_values = data.drop('Price', axis=1)

# Standardize feature values for better model performance
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_values)

# Initialize linear regression model
regression_model = LinearRegression()

# Initialize K-Fold cross-validation with 10 splits
kf = KFold(n_splits=10)

# DataFrame to store results of each fold
headers = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'RMSE', 'R2']
final_result = pd.DataFrame(columns=headers)

# Perform K-Fold cross-validation
for train_index, test_index in kf.split(x_scaled):
    # Split data into training and testing sets
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y_values.iloc[train_index], y_values.iloc[test_index]

    # Train the model
    regression_model.fit(x_train, y_train)

    # Predict target values for test set
    y_prediction = regression_model.predict(x_test)

    # Calculate evaluation metrics
    rmse = root_mean_squared_error(y_test, y_prediction)
    r2_score = regression_model.score(x_test, y_test)

    # Store coefficients and evaluation metrics in results DataFrame
    final_result.loc[len(final_result), 'feature1':'feature5'] = regression_model.coef_
    final_result.loc[len(final_result) - 1, 'RMSE'] = rmse
    final_result.loc[len(final_result) - 1, 'R2'] = r2_score

# Display results with all columns
pd.set_option('display.max_columns', None)
print(final_result)
