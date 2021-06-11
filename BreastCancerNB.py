import pandas as pd 
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
from sklearn.model_selection._validation import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import cross_val_predict


breast_cancer_df = pd.read_csv("BreastCancerData.csv")
features = ["radius_mean",	"texture_mean",	"perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]
categories = ["M", "B"]

breast_cancer_df['category'] = pd.factorize(breast_cancer_df.iloc[:,1].values)[0]
y = breast_cancer_df['category']

def make_model(x, y, df, title, x_label, y_label):
    model = GaussianNB()
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

x = breast_cancer_df.iloc[:, [2,3,6,7,8,9,10,11]]
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
    model = GaussianNB()
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

#worst feature combination is [6,11] concavity mean, texture_se 
print("worst feature combo: ",feature_combination[worst_index])
print("worst feature combo accuracy: ",feature_list_accuracy[worst_index])
heatmap_best = sns.heatmap(best_confDF, annot = True)
plt.title('Confusion Matrix for Best Set of Features')
plt.show()

#best feature combination is [2,3,6,8,10] radius_mean, texture_mean, smoothness mean, concavity mean, symmetry mean
print("best feature combo: ", feature_combination[best_index])
print("best feature combo accuracy: ",feature_list_accuracy[best_index])


#2d classification for 2 worst features 
x = breast_cancer_df.iloc[:, [6,11]].values
worst_df = pd.DataFrame(data = x)
worst_df['category'] = breast_cancer_df['category'].values
y = worst_df['category'].values
worst_df['category_name'] = breast_cancer_df.iloc[:,1].values

make_model(x,y,worst_df,'Plot with Boundaries for Worst Feature Combination', breast_cancer_df.columns[feature_combination[worst_index][0]],breast_cancer_df.columns[feature_combination[worst_index][1]])

x = breast_cancer_df.iloc[:, [2,3,6,8,10]].values
pca_best = sklearnPCA(n_components=2)
pca_best = pca_best.fit_transform(x)

best_df = pd.DataFrame(data = pca_best, columns=['eig1', 'eig2'])
best_df['category'] = breast_cancer_df['category'].values
y = best_df['category'].values
best_df['category_name'] = breast_cancer_df.iloc[:,1].values
x1 = np.array([best_df['eig1'], best_df['eig2']]).T


make_model(x1,y,best_df,'Plot with Boundaries for Best Feature Combination', 'First Eigenvector', 'Second Eigenvector')

#Naive Bayes has no parameters to tune; no hyper parameters 
# best classification we'll get is off cross validation of some sort
#570 instances, cv = 10 should yield 57 instances per bucket -> decent 

#Best Feature Cross validation 
x = breast_cancer_df.iloc[:, [2,3,6,8,10]].values
model = GaussianNB()
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
print("Best K Fold accuracy: ", accuracy)
scores = cross_validate(model, x, y, cv = 10, return_train_score = True, scoring = 'accuracy')
train_score_mean = scores['train_score'].mean()
print("training score: ", train_score_mean)

#Worst Feature Cross validation 
x = breast_cancer_df.iloc[:, [6,11]].values
model = GaussianNB()
scores = cross_val_score(model, x, y, cv = 10)
predicted = cross_val_predict(model, x, y, cv = 10)
accuracy = scores.mean()
print("Stratified accuracy worst features: ", accuracy)
confusion = confusion_matrix(y, predicted)
confDF = pd.DataFrame(data = confusion)
confDF.columns = categories
confDF.index = categories
heatmap = sns.heatmap(confDF, annot = True)
plt.title("Worst Feature Stratified K-Fold Cross Validation")
plt.show()

