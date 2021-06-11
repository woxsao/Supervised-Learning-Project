from sklearn.svm import SVC
from sklearn import svm
import pandas as pd 
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._validation import cross_val_predict
from sklearn.model_selection._validation import cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
from sklearn.preprocessing import MinMaxScaler
import json


breast_cancer_df = pd.read_csv("BreastCancerData.csv")
features = ["radius_mean",	"texture_mean",	"perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]
categories = ["M", "B"]

breast_cancer_df['category'] = pd.factorize(breast_cancer_df.iloc[:,1].values)[0]
y = breast_cancer_df['category']

def make_model(x, y, df, title, x_label, y_label):
    model = SVC()
    model = model.fit(x,y)
    xMin, xMax = x[:,0].min() - 0.1, x[:,0].max() + 0.1
    yMin, yMax = x[:,1].min() - 0.1, x[:,1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
    # np.array([xx.ravel(),yy.ravel()]).T returns a nx2 numpy array 
    z = model.predict(np.array([xx.ravel(),yy.ravel()]).T)
    z = z.reshape(xx.shape)
    plt.contourf(xx,yy,z, alpha = 0.4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    scatter = sns.scatterplot(data = df, x = df.iloc[:, 0], y = df.iloc[:, 1], hue = df.category_name.tolist(), style = df.category_name.tolist())
 
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.draw()
    plt.show()

breast_cancer_subset = breast_cancer_df.sample(n = 100)
x = breast_cancer_df.iloc[:, [2,3,6,7,8,9,10,11]]
x_sample = breast_cancer_subset.iloc[:, [2,3,6,7,8,9,10,11]]
y_sample = breast_cancer_subset['category']

feature_combination = []
temp = []
for i in range(2,9):
   temp.append(itertools.combinations([2,3,6,7,8,9,10,11],i))
for i in temp:
    for subset in i:
        feature_combination.append(subset)

predicted_model_list = []
feature_list_accuracy = []

for i in feature_combination:
    x = breast_cancer_df.iloc[:,np.array(i).astype(int)]
    y = breast_cancer_df['category'].values
    model = svm.SVC(C = 100, gamma = 'scale', kernel = 'linear')
    model = model.fit(x,y)
    predicted = model.predict(x)
    predicted_model_list.append(predicted)
    accuracy = model.score(x, y)
    feature_list_accuracy.append(accuracy)


worst_index = feature_list_accuracy.index(min(feature_list_accuracy))
best_index = feature_list_accuracy.index(max(feature_list_accuracy))


worst_predicted = predicted_model_list[worst_index]
best_predicted = predicted_model_list[best_index]

best_confusion = confusion_matrix(y, best_predicted)
best_confDF = pd.DataFrame(best_confusion)

worst_confusion = confusion_matrix(y, worst_predicted)
worst_confDF = pd.DataFrame(worst_confusion)

best_confDF.columns = categories
worst_confDF.columns = categories
best_confDF.index = categories
worst_confDF.index = categories

heatmap_worst = sns.heatmap(worst_confDF, annot = True)
plt.title('Confusion Matrix for Worst Set of Features')
plt.show()
print("worst feature combo: ",feature_combination[worst_index])
print("worst feature combo accuracy: ",feature_list_accuracy[worst_index])
heatmap_best = sns.heatmap(best_confDF, annot = True)
plt.title('Confusion Matrix for Best Set of Features')
plt.show()

print("best feature combo: ", feature_combination[best_index])
print("best feature combo accuracy: ",feature_list_accuracy[best_index])
x = breast_cancer_df.iloc[:, [2, 3, 6, 7, 8, 9]]
model = svm.SVC(C = 100, gamma = 'scale', kernel = 'linear')
scores = cross_val_score(model, x, y, cv = 10)
predicted = cross_val_predict(model, x, y, cv = 10)
accuracy = scores.mean()
print("Stratified accuracy best features: ", accuracy)
confusion = confusion_matrix(y, predicted)
confDF = pd.DataFrame(data = confusion)
confDF.columns = categories
confDF.index = categories
heatmap = sns.heatmap(confDF, annot = True)
plt.title("Best Feature Stratified K-Fold Cross Validation")
plt.show()
scores = cross_validate(model, x, y, cv = 10, return_train_score = True, scoring = 'accuracy')
train_score_mean = scores['train_score'].mean()
print("training score: ", train_score_mean)

#best is 8,9,11
"""model = svm.SVC(random_state = 0)


#excluded poly
hyperParams = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C' : [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
kFolds = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
search = GridSearchCV(model, hyperParams, cv = kFolds, scoring = 'accuracy', return_train_score = True)
#search = search.fit(x_sample, y_sample)
search = search.fit(x, y)

model = search.best_estimator_
print(search.best_params_)
#best parameters were C: 100, Gamma: scale, kernel: linear
results = pd.DataFrame(search.cv_results_)
results.to_csv("GridSearchResults.csv", header = True, index = False)



useful_results = results[['param_C', 'param_gamma', 'param_kernel', 'params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']]

useful_results['concatenated_category'] = useful_results[['param_gamma', 'param_kernel', 'param_C']].apply(lambda row: ', '.join(row.values.astype(str)), axis = 1)
print(useful_results['concatenated_category'])
useful_results['concatenated_factorized'] = pd.factorize(useful_results['concatenated_category'].values)[0]
useful_results = useful_results.sort_values(by = ['param_gamma', 'param_kernel', 'param_C'])
useful_results.to_csv("UsefulResults.csv", header = True, index = False)

x = breast_cancer_df.iloc[:, [8,9,11]]
y = breast_cancer_df['category']
c_range = [0.01, 0.1, 1, 10, 100]
parameter_list = []
trainScoresMean_list = []
trainScoresStd_list = []
testScoresMean_list = []
testScoresStd_list = []
for i in range(0,30):
    kernel_try = useful_results.iloc[i, 2]
    gamma_try = useful_results.iloc[i, 1]
    c_try = useful_results.iloc[i, 0]
    params = json.dumps(useful_results.iloc[i, 3])
    parameter_list.append(params)

    trainScoresMean = useful_results.iloc[i,6]
    trainScoresStd = useful_results.iloc[i,7]
    testScoresMean = useful_results.iloc[i,4]
    testScoresStd = useful_results.iloc[i,5]
    trainScoresMean_list.append(trainScoresMean)
    trainScoresStd_list.append(trainScoresStd)
    testScoresMean_list.append(testScoresMean)
    testScoresStd_list.append(testScoresStd)
    

    
parameter_series = pd.Series(parameter_list)
trainScoresMean_series = pd.Series(trainScoresMean_list)
trainScoresStd_series = pd.Series(trainScoresStd_list)
testScoresMean_series = pd.Series(testScoresMean_list)
testScoresStd_series = pd.Series(testScoresStd_list)
fig = plt.subplots(nrows = 1,ncols = 1, figsize = (20,10), dpi =100)
plt.plot(parameter_series, trainScoresMean_series, label = 'Training Score',\
            color = 'darkorange', lw = 1, marker = 'o', markersize = 3)
plt.fill_between(parameter_series, trainScoresMean_series - trainScoresStd_series, \
                trainScoresMean_series + trainScoresStd_series, alpha = 0.2, \
                color = 'darkorange', lw = 1)
plt.plot(parameter_series, testScoresMean_series, label = 'Validation Score', \
        color = 'navy', lw = 1, marker = 's', markersize = 3)
plt.fill_between(parameter_series, testScoresMean_series - testScoresStd_series, \
                testScoresMean_series + testScoresStd_series, alpha = 0.2, \
                color = 'navy', lw = 1)
plt.xticks(rotation = 90)
plt.legend()
plt.subplots_adjust(bottom = 0.5)
plt.show()
"""


