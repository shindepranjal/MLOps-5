from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create and train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Make prediction
prediction = model.predict([X[0]])

print("Prediction:", prediction)
print("Actual:", y[0])