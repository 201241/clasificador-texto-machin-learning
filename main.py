import sklearn as skl
# Importamos el dataset
import sklearn.datasets
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC

import csv
import pandas as pd
import numpy as np

import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

""" print(skl.__version__)
dataset = sklearn.datasets.load_iris()


# Renombramos los valores para que X sean los atributos e Y sean las respuestas del sistema
X = dataset.data
y = dataset.target

# Realizamos la partición de nuestro dataset en un conjunto de entrenamiento y otro de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Creamos el clasificador SVM lineal
classifier =  SVC(kernel="linear", C=0.025)

# Realizamos el entrenamiento
classifier.fit(X_train, y_train)

# Obtenemos el accuracy de nuestro modelo para el conjunto de test
print(classifier.score(X_test, y_test))

# Importamos la función de entrenamiento y validación cruzada
from sklearn.model_selection import cross_val_score
nScores = cross_val_score(classifier, X, y, cv=10)
# Nos devuelve un array de tipo Numpy. Podemos usar el método mean para obtener la media de los valores devueltos
print(nScores.mean()) """



toxic_or_no = pd.read_csv("dataset/youtoxic_english_1000.csv", encoding='latin-1')[["Text", "IsToxic"]]
toxic_or_no.columns = ["text", "label"]
toxic_or_no.head()

print( toxic_or_no["label"].value_counts() )


punctuation = set(string.punctuation)

def tokenize(sentence):
    tokens = []
    for token in sentence.split():
        new_token = []
        for character in token:
            if character not in punctuation:
                new_token.append(character.lower())
        if new_token:
            tokens.append("".join(new_token))
    return tokens

toxic_or_no.head()["text"].apply(tokenize)

demo_vectorizer = CountVectorizer(
    tokenizer = tokenize,
    binary = True
)


train_text, test_text, train_labels, test_labels = train_test_split(toxic_or_no["text"], toxic_or_no["label"], stratify=toxic_or_no["label"])
print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")

real_vectorizer = CountVectorizer(tokenizer = tokenize, binary=True)
train_X = real_vectorizer.fit_transform(train_text)
test_X = real_vectorizer.transform(test_text)


classifier = LinearSVC()
classifier.fit(train_X, train_labels)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)

predicciones = classifier.predict(test_X)
accuracy = accuracy_score(test_labels, predicciones)
print(f"Accuracy: {accuracy:.2%}")

frases = [
    'the president and owner of the channel are idiots',
    'black people shouldn\'t go into those stores',
    'police are paid to kill',
    'good video keep it up',
    'you helped me a lot bro',
    'I wait for the next video'
]

frases_X = real_vectorizer.transform(frases)
predicciones = classifier.predict(frases_X)

print("\n\n¿Es un comentario toxico?\n")
for text, label in zip(frases, predicciones): 
    print(f"{label} - {text}")

print("\n\nfin...")