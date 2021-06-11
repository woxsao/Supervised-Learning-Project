import pandas as pd 
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
from sklearn.model_selection._validation import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


breast_cancer_df = pd.read_csv("BreastCancerData.csv")
features = ["radius_mean",	"texture_mean",	"perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]
categories = ["M", "B"]



#have to normalize data first before running tests, 
#(xi-min(x))/(max(x)-min(x))
new_bc_df = pd.DataFrame()
new_bc_df['id'] = breast_cancer_df['id']
new_bc_df['diagnosis'] = breast_cancer_df['diagnosis']

#Normalizing Data
for i in range(2, 32):
    temp = breast_cancer_df.iloc[:, i].astype(float)
    temp_min = temp.min()
    temp_max = temp.max()
    temp_final = (temp-temp_min)/(temp_max-temp_min)
    new_bc_df[breast_cancer_df.columns[i]] = temp_final


new_bc_df.to_csv('NormalizedBCData.csv', index = False, header = True)   
new_bc_df['category'] = pd.factorize(new_bc_df.iloc[:,1].values)[0]
y = new_bc_df['category']

def make_model(x, y, df, title, x_label, y_label):
    model = KNeighborsClassifier(n_neighbors = 19, metric = 'manhattan', weights = 'uniform')
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

x = new_bc_df.iloc[:, [2,3,6,7,8,9,10,11]]
feature_combination = []
temp = []
for i in range(2,9):
   temp.append(itertools.combinations([2,3,6,7,8,9,10,11],i))
for i in temp:
    for subset in i:
        feature_combination.append(subset)

predicted_model_list = []
feature_list_accuracy = []




x = new_bc_df.iloc[:, [2,3,6,7,8,9,10,11]]
pca_best = sklearnPCA(n_components=2)
pca_best = pca_best.fit_transform(x)

best_df = pd.DataFrame(data = pca_best, columns=['eig1', 'eig2'])
best_df['category'] = new_bc_df['category'].values
y = best_df['category'].values
best_df['category_name'] = new_bc_df.iloc[:,1].values
x1 = np.array([best_df['eig1'], best_df['eig2']]).T


make_model(x1,y,best_df,'Plot with Boundaries for ALL Features', 'First Eigenvector', 'Second Eigenvector')



for i in feature_combination:
    x = new_bc_df.iloc[:,np.array(i).astype(int)]
    y = new_bc_df['category'].values
    model = KNeighborsClassifier()
    model = model.fit(x,y)
    predicted = model.predict(x)
    predicted_model_list.append(predicted)
    accuracy = model.score(x, y)
    feature_list_accuracy.append(accuracy)

feature_comboDF = pd.DataFrame(data = [feature_combination,feature_list_accuracy])
feature_comboDF.to_csv('featureComboDFBreast.csv', header = False, index = False)

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

#worst feature combination is [6,10] concavity mean, symmetry_mean 
print("worst feature combo: ",feature_combination[worst_index])
print("worst feature combo accuracy: ",feature_list_accuracy[worst_index])
heatmap_best = sns.heatmap(best_confDF, annot = True)
plt.title('Confusion Matrix for Best Set of Features')
plt.show()

#best feature combination is [2,3,6,8,10,11] radius_mean, texture_mean, smoothness mean, concavity mean, symmetry mean, fractal_dimension_mean
print("best feature combo: ", feature_combination[best_index])
print("best feature combo accuracy: ",feature_list_accuracy[best_index])


#2d classification for 2 worst features 
x = new_bc_df.iloc[:, [6,10]].values
worst_df = pd.DataFrame(data = x)
worst_df['category'] = new_bc_df['category'].values
y = worst_df['category'].values
worst_df['category_name'] = new_bc_df.iloc[:,1].values

make_model(x,y,worst_df,'Plot with Boundaries for Worst Feature Combination', new_bc_df.columns[feature_combination[worst_index][0]],new_bc_df.columns[feature_combination[worst_index][1]])

x = new_bc_df.iloc[:, [2, 3, 6, 8, 10, 11]].values
pca_best = sklearnPCA(n_components=2)
pca_best = pca_best.fit_transform(x)

best_df = pd.DataFrame(data = pca_best, columns=['eig1', 'eig2'])
best_df['category'] = new_bc_df['category'].values
y = best_df['category'].values
best_df['category_name'] = new_bc_df.iloc[:,1].values
x1 = np.array([best_df['eig1'], best_df['eig2']]).T


make_model(x1,y,best_df,'Plot with Boundaries for Best Feature Combination', 'First Eigenvector', 'Second Eigenvector')


x = new_bc_df.iloc[:, [2,3,6,8,10, 11]].values
model = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 25, weights = 'uniform')
scores = cross_val_score(model, x, y, cv = 10)
predicted = cross_val_predict(model, x, y, cv = 10)
accuracy = scores.mean()
confusion = confusion_matrix(y, predicted)
confDF = pd.DataFrame(data = confusion)
confDF.columns = categories
confDF.index = categories
heatmap = sns.heatmap(confDF, annot = True)
plt.title("Confusion Matrix for 10-Fold Cross Validation")
plt.show()
print(accuracy)

scores = cross_validate(model, x, y, cv = 10, return_train_score = True, scoring = 'accuracy')
train_score_mean = scores['train_score'].mean()
print("training score: ", train_score_mean)

x = breast_cancer_df.iloc[:, [2,3,6,7,8,9,10,11]]

model = KNeighborsClassifier()
maxNeighborRange = range(1,30)
trainScores, testScores = validation_curve(model, x, y,
            param_name = 'n_neighbors', param_range = maxNeighborRange,
            cv = 10, scoring = 'accuracy')

trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores,axis = 1)
testScoresMean = np.mean(testScores, axis = 1)
testScoresStd = np.std(testScores, axis = 1)


plt.plot(maxNeighborRange, trainScoresMean, label = 'Training Score', \
         color = 'darkorange', lw = 1, marker = 'o', markersize = 3)
plt.fill_between(maxNeighborRange, trainScoresMean - trainScoresStd, \
                 trainScoresMean + trainScoresStd, alpha = 0.2, \
                 color = 'darkorange', lw = 1)
plt.plot(maxNeighborRange, testScoresMean, label = 'Validation Score', \
         color = 'navy', lw = 1, marker = 's', markersize = 3)
plt.fill_between(maxNeighborRange, testScoresMean - testScoresStd, \
                 testScoresMean + testScoresStd, alpha = 0.2, \
                 color = 'navy', lw = 1)
plt.legend()
plt.title("Validation Curve tuning n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.show()

hyperParams = {'weights': ['uniform', 'distance'],
                'n_neighbors': range(5,30), 
                'metric': ['euclidean', 'manhattan']}
kFolds = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)
search = GridSearchCV(model, hyperParams, cv = kFolds, scoring = 'accuracy')
search = search.fit(x, y)

model = search.best_estimator_
print(search.best_params_)





