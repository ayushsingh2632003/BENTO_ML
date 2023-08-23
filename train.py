import bentoml

from sklearn import svm
from sklearn import datasets


#load training datasets
iris=datasets.load_iris()
X,y=iris.data,iris.target


#train the model
clf=svm.SVC(gamma='scale')

clf.fit(X,y)

#saved teh mdoel to the BentoML local model store 
saved_model=bentoml.sklearn.save_model("iris_clf",clf)

print(f"Model saved :{saved_model}")


#  iris_clf:e4h66akbvgnhn3bo