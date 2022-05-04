import pandas as pd
import numpy as np
import sklearn
from matplotlib import style
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot


# 1. Collecting the Data #
data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# 2. Preparing the Data #
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "Medu"]]  # Selecting columns to use
print(data.head())

predict = "G3"  # Selecting the attribute to predict

X = np.array(data.drop([predict], 1))  # Creating Label Instances Array
y = np.array(data[predict])  # Creating Prediction Instances Array

# Splitting test and train dataset
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

"""best = 0
for _ in range(5000):  # Choosing the best Accuracy out of 50 (range) Test-Training Set Variation
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) # Resplitting
    # 3. Choosing a Model #
    model = linear_model.LinearRegression()  # Choosing Linear Regression Model

    # 4. Training the Model #
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)  # Evaluating the Model
    print(acc)

    if acc > best:
        best = acc
        # 4.1 Saving the Model for Future
        with open("studentmodel.pickle", "wb") as f:  # Write the pickled rep. of object to the open file object file
            pickle.dump(model, f)"""

# print('best:', best)
pickle_in = open("studentmodel.pickle", "rb")  # Read the pickled rep. of an object from the open file object file
model = pickle.load(pickle_in)

# 5. Evaluating the Model #
print('Accuracy = ', model.score(x_test, y_test))
# print(linear.coef_)
# print(linear.intercept_)

# 6. Parameter Tuning #

# 7. Making Predictions #
predictions = model.predict(x_test)
for x in range(10):
    print('\nPredicted Final Note:', predictions[x], '\nFactors:', x_test[x], '\nActual Final Note:', y_test[x])


# Playground
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


