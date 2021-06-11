import pandas as pd 
import itertools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
from sklearn.model_selection._validation import cross_val_predict, cross_validate



breast_cancer_df = pd.read_csv("BreastCancerData.csv")
features = ["radius_mean",	"texture_mean",	"perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]
categories = ["M", "B"]

breast_cancer_df['category'] = pd.factorize(breast_cancer_df.iloc[:,1].values)[0]
y = breast_cancer_df['category']
"""

def make_model(x, y, df, title, x_label, y_label):
    model = DecisionTreeClassifier()
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


#2d classification for 2 features

x = breast_cancer_df.iloc[:, 2:4].values
print(x)
worst_df = pd.DataFrame(data = x)
worst_df['category'] = breast_cancer_df['category']
worst_df['category_name'] = breast_cancer_df.iloc[:,1]

make_model(x,y,worst_df,'Plot with Boundaries for Best? Feature Combination', "radius_mean", "texture_mean")



"""
"""
#Projecting everything
breast_cancer_df = breast_cancer_df.reset_index()
breast_cancer_df = breast_cancer_df.fillna(0)

x = breast_cancer_df.iloc[:,3:].apply(pd.to_numeric)
y = breast_cancer_df['category'].values
model = DecisionTreeClassifier()
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
"""



"""x = breast_cancer_df.iloc[:, 2:32].apply(pd.to_numeric)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
model = DecisionTreeClassifier(criterion="entropy", max_depth = None, random_state=0)
model.fit(x_train,y_train)

score = cross_val_score(model, x,y, cv = 10, scoring = "accuracy").mean()
print(score)
fig = plt.subplots(nrows = 1,ncols = 1,figsize = (50,30), dpi = 40)
plot_tree(model, filled = True, fontsize = 10, feature_names=breast_cancer_df.columns, class_names=categories)
plt.show()
plt.savefig("DecisionTree.png")"""
"""
Comments: 
After trying multiple feature combinations the validation score is hovering around the same area

Try PCA with maxdepth = 5 on the features instead of max depth = none
Try max depth 5 to find best feature combination
"""


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
    model = DecisionTreeClassifier(max_depth = 5)
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
print("worst feature combo: ",feature_combination[worst_index])
print("worst feature combo accuracy: ",feature_list_accuracy[worst_index])
heatmap_best = sns.heatmap(best_confDF, annot = True)
plt.title('Confusion Matrix for Best Set of Features')
plt.show()

print("best feature combo: ", feature_combination[best_index])
print("best feature combo accuracy: ",feature_list_accuracy[best_index])

"""
def make_model(x, y, df, title, x_label, y_label):
    model = DecisionTreeClassifier(max_depth=5)
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


#2d classification for 2 features
x = breast_cancer_df.iloc[:, [6,10]].values
worst_df = pd.DataFrame(data = x)
worst_df['category'] = breast_cancer_df['category'].values
y = worst_df['category'].values
worst_df['category_name'] = breast_cancer_df.iloc[:,1].values

make_model(x,y,worst_df,'Plot with Boundaries for Worst Feature Combination', breast_cancer_df.columns[feature_combination[worst_index][0]],breast_cancer_df.columns[feature_combination[worst_index][1]])

x = breast_cancer_df.iloc[:, [2,3,6,7,8,9,10,11]].values
pca_best = sklearnPCA(n_components=2)
pca_best = pca_best.fit_transform(x)

best_df = pd.DataFrame(data = pca_best, columns=['eig1', 'eig2'])
best_df['category'] = breast_cancer_df['category'].values
y = best_df['category'].values
best_df['category_name'] = breast_cancer_df.iloc[:,1].values
x1 = np.array([best_df['eig1'], best_df['eig2']]).T


make_model(x1,y,best_df,'Plot with Boundaries for Best Feature Combination', 'First Eigenvector', 'Second Eigenvector')
"""


x = breast_cancer_df.iloc[:, [2,3,6,7,8,9,10,11]]

model = DecisionTreeClassifier()
maxDepthRange = range(1, 30)
trainScores, testScores = validation_curve(model, x, y,
            param_name = 'max_depth', param_range = maxDepthRange, 
            cv = 10, scoring = 'accuracy')

trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores,axis = 1)
testScoresMean = np.mean(testScores, axis = 1)
testScoresStd = np.std(testScores, axis = 1)
print("test scores: " , testScores)
print("test scores mean: ", testScoresMean)
print("test scores std: ", testScoresStd)

plt.plot(maxDepthRange, trainScoresMean, label = 'Training Score', \
         color = 'darkorange', lw = 1, marker = 'o', markersize = 3)
plt.fill_between(maxDepthRange, trainScoresMean - trainScoresStd, \
                 trainScoresMean + trainScoresStd, alpha = 0.2, \
                 color = 'darkorange', lw = 1)
plt.plot(maxDepthRange, testScoresMean, label = 'Validation Score', \
         color = 'navy', lw = 1, marker = 's', markersize = 3)
plt.fill_between(maxDepthRange, testScoresMean - testScoresStd, \
                 testScoresMean + testScoresStd, alpha = 0.2, \
                 color = 'navy', lw = 1)
plt.title("Validation Curve tuning Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




model = DecisionTreeClassifier(random_state = 0)
# Set up the grid of parameters
hyperParams = {'criterion': ['gini', 'entropy'],
               'max_depth': range(5, 15)}
# Train the model in the grid search
kFolds = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)
search = GridSearchCV(model, hyperParams, cv = kFolds, scoring = 'accuracy')
search = search.fit(x, y)
# Show some of the results using a dataframe

model = search.best_estimator_
print(search.best_params_)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
model = DecisionTreeClassifier(criterion="entropy", max_depth = 4, random_state=0)
model.fit(x_train,y_train)

score = cross_val_score(model, x,y, cv = 10, scoring = "accuracy").mean()
print(score)



model = DecisionTreeClassifier(random_state = 0)
hyperParams = {'criterion': ['gini', 'entropy'],
               'max_depth': range(2, 5)}
search = GridSearchCV(model, hyperParams, cv = 2, scoring = 'accuracy')
scores = cross_val_score(search, x, y, scoring = 'accuracy', cv = 5)
print("scores: ", scores)
print('Accuracy:', np.mean(scores), ' +/- ', np.std(scores))
predicted = cross_val_predict(search, x, y, cv = 5)
confusion = confusion_matrix(y, predicted)
grid_search_df = pd.DataFrame(confusion)
grid_search_df.index = categories
grid_search_df.columns = categories
GridSearchConfusion = sns.heatmap(grid_search_df, annot = True)
plt.title('Confusion Matrix for Decision Tree classification')
plt.show()

model = DecisionTreeClassifier(random_state=0, max_depth=5, criterion="entropy")
scores = cross_val_score(model, x, y, cv = 10)
predicted = cross_val_predict(model, x, y, cv = 10)
accuracy = scores.mean()
confusion_best_parameters = confusion_matrix(y, predicted)
confusion_best_parameters_df = pd.DataFrame(confusion_best_parameters)
confusion_best_parameters_df.index = categories
confusion_best_parameters_df.columns = categories
BestParam_confusion = sns.heatmap(confusion_best_parameters_df, annot= True)
plt.title('Confusion Matrix for Decision Tree Best Params')
plt.show()
print("Accuracy for My test: ", accuracy)

x = breast_cancer_df.iloc[:, [2,3,6,7,8, 9,11]]
model = DecisionTreeClassifier(random_state=0, max_depth=5, criterion="entropy")
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
