import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.float_format', '{:.4f}'.format)

df = pd.read_excel("data/DryBeanDataSet.xlsx")

df['Extent'] = df['Extent'].replace("?", np.nan)
df['Compactness'] = df['Compactness'].replace("?", np.nan)
df['ShapeFactor6'] = df['ShapeFactor6'].replace("?", np.nan)

df = df[~df['Extent'].isnull()]
df = df[~df['Compactness'].isnull()]
df = df[~df['ShapeFactor6'].isnull()]
df = df[df['Colour'] != "?"]
df = df[df['Class'] != "?"]
df = df.drop(['Sort order'], axis=1)

df = df[df['EquivDiameter'] < 1000]
df = df[df['ConvexArea'] > 0]

to_scale = [
    'Area',
    'Perimeter',
    'MajorAxisLength',
    'MinorAxisLength',
    'Eccentricity',
    'ConvexArea',
    'EquivDiameter',
    'ShapeFactor6',
    'Extent',
    'Solidity',
    'roundness',
    'Compactness',
    'ShapeFactor1',
    'ShapeFactor2',
    'ShapeFactor3',
    'ShapeFactor4',
    'ShapeFactor5'
]

X = df[to_scale]

# KNN
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=to_scale, index=df.index)

X['AspectRation'] = df['AspectRation']
X['Constantness'] = df['Constantness']

encoder = LabelEncoder()
X['Colour'] = encoder.fit_transform(df['Colour'])
encoder = LabelEncoder()
X['Class'] = encoder.fit_transform(df['Class'])

X.to_csv("data/knn.csv", index=False)

encoder = LabelEncoder()
df['Colour'] = encoder.fit_transform(df['Colour'])
encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])

df.to_csv("data/dt.csv", index=False)



