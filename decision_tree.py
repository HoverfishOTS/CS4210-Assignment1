#-------------------------------------------------------------------------
# AUTHOR: Ryan Wei
# FILENAME: decision_tree.py
# SPECIFICATION: This program implements a decision tree classifier for predicting contact lens suitability.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
X = []
for row in db:
    feature_row = [
        0 if row[0] == 'Young' else 1 if row[0] == 'Prepresbyopic' else 2,
        0 if row[1] == 'Myope' else 1,
        0 if row[2] == 'No' else 1,
        0 if row[3] == 'Reduced' else 1
    ]
    X.append(feature_row)

#encode the original categorical training classes into numbers and add to the vector Y.
Y = [0 if row[4] == 'Yes' else 1 for row in db]

#fitting the depth-2 decision tree to the data using entropy as your impurity measure
clf = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf = clf.fit(X, Y)

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()