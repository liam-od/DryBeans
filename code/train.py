import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
import math

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['text.antialiased'] = True
pd.set_option('display.float_format', '{:.4f}'.format)

# KNN
df = pd.read_csv("data/knn.csv")

# Tree
df = pd.read_csv("data/dt.csv")


cols = df.columns
target = cols[-1]
features = cols[:-1]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sco = []
iters = []
for i in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=i)

    f1_scorer = make_scorer(f1_score, average='macro')
    cv_results = cross_validate(model, X_test, y_test, cv=5, scoring=f1_scorer, return_estimator=True)

    scores = cv_results['test_score']
    print(f"({i}) Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    sco.append(scores.mean())
    iters.append(i)

plt.figure(figsize=(5, 4))
plt.plot(iters, sco)
plt.title("Choosing n_neighbours")
plt.xlabel("n_neighbours")
plt.ylabel("f1 macro")
plt.savefig("neigh.pdf", format='pdf')


fitted = cv_results['estimator'][0]

# Test
y_pred = fitted.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 score on test set: {f1_macro:.3f}")

f1_per_class = f1_score(y_test, y_pred, average=None)
print(f1_per_class)


# Classification tree
sco = []
iters = []
for i in range(2, 100):
    model = DecisionTreeClassifier(min_samples_split=i)
    f1_scorer = make_scorer(f1_score, average='macro')
    cv_results = cross_validate(model, X_test, y_test, cv=5, scoring=f1_scorer, return_estimator=True)
    scores = cv_results['test_score']
    print(f"({i}) Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    sco.append(scores.mean())
    iters.append(i)

plt.figure(figsize=(5, 4))
plt.plot(iters, sco)
plt.title("Choosing max_samples_split")
plt.xlabel("max_samples_split")
plt.ylabel("f1 macro")
plt.savefig("samples.pdf", format='pdf')

# Test
y_pred = fitted.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 score on test set: {f1_macro:.3f}")

f1_per_class = f1_score(y_test, y_pred, average=None)
print(f1_per_class)







