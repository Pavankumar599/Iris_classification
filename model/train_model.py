from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a model
model = RandomForestClassifier()
model.fit(X,y)

# Save the model
with open('model/Iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved  successfully")