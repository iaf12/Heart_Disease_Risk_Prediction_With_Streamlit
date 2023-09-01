import pandas as pd
from sklearn import tree

df = pd.read_csv('CVD_newfile.csv')
df.drop(columns=df.columns[0], axis=1, inplace=True)
numerical = df.select_dtypes(include=['float64']).columns.sort_values()
categorical = df.select_dtypes(include=['object']).columns.sort_values()
categorical=categorical.drop('Heart_Disease')
# df.drop(['Height_(cm)','Weight_(kg)'], axis=1, inplace=True)

for col in categorical:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'No':0, 'Yes':1}
def target_encode(val):
    return target_mapper[val]

df['Heart_Disease'] = df['Heart_Disease'].apply(target_encode)

x = df.drop('Heart_Disease', axis=1)
y = df['Heart_Disease']


model = tree.DecisionTreeClassifier()
model.fit(x, y)

# Saving the model
import pickle
pickle.dump(model, open('heart_model_1.pkl', 'wb'))
