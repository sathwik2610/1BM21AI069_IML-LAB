from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

with open('training.json') as file:
    lines = file.readlines()
lines = iter(lines)

N = int(next(lines))

X_train, classes = [], []
for _ in range(N):
    line = next(lines)
    d = eval(line)
    X_train.append(d['question'])
    classes.append(d['topic'].strip())

classes_to_ix = {c: i for i, c in enumerate(set(classes))}
ix_to_classes = {i: c for c, i in classes_to_ix.items()}
y_train = [classes_to_ix[c] for c in classes]

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")
X_train = vectorizer.fit_transform(X_train)

clf = MultinomialNB(alpha=0.1)
clf.fit(X_train, y_train)

P = int(input())
X_pred = []
for _ in range(P):
    line = input()
    d = eval(line)
    X_pred.append(d['question'])
    
X_pred = vectorizer.transform(X_pred)
y_pred = [ix_to_classes[i] for i in clf.predict(X_pred)]
print('\n'.join(y_pred))
