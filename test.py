import pandas as pd
from sklearn.model_selection import train_test_split
from classifion import GoClassify
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/test.csv')

y = data['trigger_flag']
X = data.drop('trigger_flag', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size = 0.3)

clf = GoClassify()
best = clf.train(x_train, y_train)
y_pred = best.predict(x_test)
print(f'accuracy: {accuracy_score(y_pred,y_test)}')
