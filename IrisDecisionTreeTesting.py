import pandas as pd 
import itertools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._validation import cross_val_predict



iris_df = pd.read_csv("Iris.csv")
categories = ['Setosa', 'Versicolor', 'Virginica']
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
iris_df['category'] = pd.factorize(iris_df.iloc[:,4].values)[0]

temp = []
#change range to 2,31
for i in range(2, 5):
    temp.append(np.array(list(itertools.combinations(range(0,4), i)))) #change 0,4 -> 2, 33


feature_combination = []
for i in temp:
    for j in i:
        feature_combination.append(j)


#Decision Tree Classifier
y = iris_df['category'].values

predicted_model_list = []
feature_list_accuracy = []

for i in feature_combination:
    x = iris_df.iloc[:,np.array(i).astype(int)]
    y = iris_df['category'].values
    model = DecisionTreeClassifier()
    model = model.fit(x,y)
    predicted = model.predict(x)
    predicted_model_list.append(predicted)
    accuracy = model.score(x, y)
    feature_list_accuracy.append(accuracy)

feature_comboDF = pd.DataFrame(data = [feature_combination,feature_list_accuracy])
feature_comboDF.to_csv('featureComboDF.csv', header = False, index = False)

#Indices to isolate the terms
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
heatmap_best = sns.heatmap(best_confDF, annot = True)
plt.title('Confusion Matrix for Best Set of Features')
plt.show()





def make_model(x, y, df, title, x_label, y_label):
    model = DecisionTreeClassifier()
    model = model.fit(x,y)
    xMin, xMax = x[:,0].min() - 0.1, x[:,0].max() + 0.1
    yMin, yMax = x[:,1].min() - 0.1, x[:,1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
    # np.array([xx.ravel(),yy.ravel()]).T returns a nx2 numpy array 
    print(xx) 
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
x = iris_df.iloc[:,feature_combination[worst_index]].values
worst_df = pd.DataFrame(data = x)
worst_df['category'] = iris_df['category']
worst_df['category_name'] = iris_df.iloc[:,4]

make_model(x,y,worst_df,'Plot with Boundaries for Worst Feature Combination', features[0], features[1])

#2d classification in PCA Space for 3 features
x = iris_df.iloc[:, feature_combination[best_index]].values
pca = sklearnPCA(n_components= 2)
pca_projection = pca.fit_transform(x)
irisPCA_df = pd.DataFrame(data = pca_projection, columns = ['eig1', 'eig2'])
irisPCA_df['category'] = iris_df['category']
irisPCA_df['category_name'] = iris_df.iloc[:,4]

x1 = np.array([irisPCA_df['eig1'], irisPCA_df['eig2']]).T

model = DecisionTreeClassifier()
#fitting model to eigenvectors 
model = model.fit(x,y)

xMin, xMax = x1[:,0].min() - 0.1, x1[:,0].max() + 0.1
yMin, yMax = x1[:,1].min() - 0.1, x1[:,1].max() + 0.1

xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))

z = model.predict(pca.inverse_transform(np.array([xx.ravel(), yy.ravel()]).T))
#Therefore we need this reshape step in order to condense it back into our desired parameters. 
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, alpha = 0.4)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
scatter_pca = sns.scatterplot(data = irisPCA_df, x = 'eig1', y = 'eig2', hue = irisPCA_df.category_name.tolist(), style = irisPCA_df.category_name.tolist())

plt.title('PCA Plot with Boundaries for Best Feature Combination')
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.draw()
plt.show()


#Projecting everything

x = iris_df.iloc[:,:-2]
pca = sklearnPCA(n_components=2)
pca_projection = pca.fit_transform(x)

irisPCA_df = pd.DataFrame(data = pca_projection, columns=['eig1', 'eig2'])
irisPCA_df['category'] = iris_df['category']
irisPCA_df['category_name'] = iris_df.iloc[:,4]
x1 = np.array([irisPCA_df['eig1'], irisPCA_df['eig2']]).T
make_model(x1, y, irisPCA_df, 'Iris Projected and Classified in PCA Space', 'eig1', 'eig2')


x = iris_df.iloc[:,:-2]
y = iris_df['category'].values
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


model = DecisionTreeClassifier()
x = iris_df.iloc[:, :-2]
model = model.fit(x,y)

fig = plt.subplots(nrows = 1,ncols = 1,figsize = (30,30), dpi = 60)
plot_tree(model, filled = True, fontsize = 10)
plt.show()
plt.savefig("DecisionTree.png")
