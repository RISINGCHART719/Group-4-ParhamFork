#I chose the dataset and preprocessing method from hw2 (which was KNN).
import numpy as np
import pandas as pandaslib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



#reading the dataset using pandas.
#Dataset has data about certain dates and whether it rained
# or not on the following day in Australia.
data = pandaslib.read_csv('weatherAUS.csv')

#Extracting relevant features.
# *Date does not matter much
# *Location does not matter much, especially because we are
#    also dropping WindGustDir, WindDir9am, and WindDir3pm
# *WindGustDir, *WindDir9am, and *WindDir3pm are somewhat important, but in hw2,
#    I did not do any label encoding. In order to keep the preprocessing the same,
#    I will drop these features and not label encode them.
# *RainToday is dropped because it is just a binary value that is true when the
#    Rainfall feature is more than 1.
data.drop( ['Date', 'Location', 'WindGustDir', 'WindDir9am' , 'WindDir3pm', 'RainToday'],axis = 1, inplace = True)


#replacing any value that is labeled as 'NA' in the file with an actual empty value
data.replace('NA', np.nan, inplace=True)
#dropping the samples that have empty in them. We can afford to do this
#instead of imputing the rows that have missing values
# because there are many full rows (about 58000).
data.dropna(inplace=True)


#extracting the dependent variable (binary value)
yValues = data['RainTomorrow']

#extracting independent variables. They are all interpreted as
# floats. RainTomorrow is the classification (whether it rains tomorrow or not),
# and so it is not included in our independent variables.
xValues = data.drop('RainTomorrow', axis = 1)

#in hw2, I normalized the data for KNN. In order to keep the preprocessing the same,
#I will do the same for this assignment.
scalerInst = StandardScaler()
xValues = scalerInst.fit_transform(xValues)


#splitting our data into the test and the train subsets
#Train for training our model, and test for testing the acquired model.
trainX, testX, trainY, testY = train_test_split(xValues, yValues, test_size=0.25)

# Create the MLP classifier model
#4 hidden layers, each with 30 nodes. Using logistic activation which is sigmoid function.
# Optimizer is adam, which is adaptive moment estimation, that adjusts the learning rate based on
# past gradients. It is better for larger datasets (like the one for this assignment), since it makes
# the model faster. I also used an adaptive learning rate, since it improves convergence speed.
MLPModel = MLPClassifier(hidden_layer_sizes=(40,40,40), activation ='logistic',
                         solver = 'adam', alpha = 0.001, batch_size= 'auto', learning_rate='adaptive',
                         shuffle = True, max_iter=1000, random_state=100)

#training the model
MLPModel.fit(trainX, trainY)

#calculating the accuracy score and printing it.
#****NOTE: since the model shuffles the data each time, the calculated
#accuracy will be a bit different from the reported accuracy in the report file.
score = MLPModel.score(testX, testY)
print('MLP Model\'s accuracy: ', score * 100 , '%')