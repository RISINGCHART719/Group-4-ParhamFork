import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv('housing_price_dataset.csv')

#convert to nominal
le = LabelEncoder()
label = le.fit_transform(data['Neighborhood'])

#replace with int
data.drop("Neighborhood", axis=1, inplace=True)
data["Neighborhood"] = label

yValues = data['Price']
xValues = data.drop('Price', axis = 1)
scalerInst = StandardScaler()
xScaled = scalerInst.fit_transform(xValues)

regressionModel = LinearRegression()

#kfold
kf = KFold(n_splits=10)

#final result array
headers = ['feature1', 'feature2', 'feature3','feature4','feature5', 'RMSE', 'R2']
finalResult = pd.DataFrame(columns=headers)


#calculating the features' coefficients and RMSE and R2 (score) for each of the folds
#in the following loop
for trainIndex, testIndex in kf.split(xScaled):
    xTrain, xTest = xScaled[trainIndex], xScaled[testIndex]
    yTrain, yTest = yValues[trainIndex], yValues[testIndex]

    #training the model
    regressionModel.fit(xTrain, yTrain)

    #predicting and testing accuracy
    yPrediction = regressionModel.predict(xTest)
    rootMeanSquareError = root_mean_squared_error(yTest, yPrediction)
    finalResult.loc[len(finalResult), 'feature1':'feature5'] = regressionModel.coef_
    finalResult.loc[len(finalResult) - 1, 'RMSE'] = rootMeanSquareError
    finalResult.loc[len(finalResult) - 1, 'R2'] = regressionModel.score(xTest, yTest)


#printing the results
pd.set_option('display.max_columns', None)
print(finalResult)