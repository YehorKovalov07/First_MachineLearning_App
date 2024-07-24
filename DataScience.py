import pandas as pd

from sklearn.tree import DecisionTreeClassifier

#from sklearn.model_selection import train_test_split

music_data = pd.read_csv('musicData.csv')

X = music_data.drop(columns=['genre'])

Y  = music_data['genre']

model = DecisionTreeClassifier()

model.fit(X,Y)

predictions = model.predict([[21, 1], [21,0]])

print(predictions)