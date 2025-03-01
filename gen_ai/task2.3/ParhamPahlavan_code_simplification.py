import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

#reading in the data
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
le = LabelEncoder()

#converting the categorical features to numerical
categoricalFeatures = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS' ]
for column in categoricalFeatures:
    data[column] = le.fit_transform(data[column])

#separating the data into dependent and independent sub-groups.
yValues = data['NObeyesdad']
xValues = data.drop('NObeyesdad', axis = 1)
xValues = xValues.to_numpy()

#creating the random forest model.
classifierModel = RandomForestClassifier(n_estimators=90, random_state=20, max_depth=25)

#Creating PCA with 2 dimensions
pca = PCA(n_components= 2)

#creating a color-mapping and a legend for the scatter plots
weight_colors = {'Normal_Weight':'r', 'Overweight_Level_I':'b', 'Overweight_Level_II':'g', 'Obesity_Type_I':'orange', 'Obesity_Type_II':'purple', 'Obesity_Type_III':'black','Insufficient_Weight':'pink' }
legendHandles = [
    mpatches.Patch(color=weight_colors['Normal_Weight'], label='Normal Weight'),
    mpatches.Patch(color=weight_colors['Overweight_Level_I'], label='Overweight_Level_I'),
    mpatches.Patch(color=weight_colors['Overweight_Level_II'], label='Overweight_Level_II'),
    mpatches.Patch(color=weight_colors['Obesity_Type_I'], label='Obesity_Type_I'),
    mpatches.Patch(color=weight_colors['Obesity_Type_II'], label='Obesity_Type_II'),
    mpatches.Patch(color=weight_colors['Obesity_Type_III'], label='Obesity_Type_III'),
    mpatches.Patch(color=weight_colors['Insufficient_Weight'], label='Insufficient_Weight')
]

#Making the folds.
KF = KFold(n_splits = 10, shuffle = True, random_state = 95)

#making the array to hold the final results, as well as
#a variable that is used to store the results.
finalFoldResults = pd.DataFrame()
tempIndex = 0
for trainIndex, testIndex in KF.split(xValues):
    xTrain, xTest = xValues[trainIndex], xValues[testIndex]
    yTrain, yTest = yValues[trainIndex], yValues[testIndex]

#fitting the model based on the fold's train data.
    classifierModel.fit(xTrain, yTrain)

#calculating and saving the results into the final array.
    finalFoldResults.loc[1, tempIndex] = classifierModel.score(xTest, yTest)
    tempIndex += 1

#fitting the data for this fold into the pca
    xTrain_pca = pca.fit_transform(xTrain)
    plt.figure(figsize = (10,10))
    colors = [weight_colors[label] for label in yTrain]
    plt.scatter(xTrain_pca[:, 0], xTrain_pca[:, 1], c = colors, alpha = 0.7)
    plt.title('PCA Training Data Visualization For Fold %d' % tempIndex)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(handles = legendHandles, title = "Weights")
    plt.show()

foldHeaders = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
finalFoldResults.columns = [foldHeaders]

#printing the final results.
pd.set_option('display.max_columns', None)
print(finalFoldResults)

#representing the results visually
tempResults = finalFoldResults.iloc[0].values.tolist()
plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy of Folds')
plt.ylim(0.5, 1)
plt.bar_label(plt.bar(foldHeaders,tempResults,width = 0.65, color=  'maroon'), labels= [f'{t:.4f}' for t in tempResults], fontsize = 8)
plt.show()


#calculating the average fold accuracy
averageFoldAccuracy = sum(tempResults) / len(tempResults)
print('Average Fold Accuracy: ', averageFoldAccuracy)