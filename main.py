import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PATH = "C:/Users/ecrvd/PycharmProjects/DiseasePrediction/"
DATA_DISEASE = "dataset.csv"
DATA_SEV = "symptom-severity.csv"
DATASET = pd.read_csv(PATH + DATA_DISEASE)
SEVERITY = pd.read_csv(PATH + DATA_SEV)
df = DATASET.copy()

filename1 = "Models/lr_model.sav"
filename2 = "Models/dt_model.sav"
filename3 = "Models/sgd_model.sav"
filename4 = "Models/rf_model.sav"
file_symp = "Models/symptoms.p"

os.makedirs("Models/", exist_ok=True)

if os.path.exists(file_symp):
    print("Symptoms existe")
else:
    symptoms = df.iloc[:, 1:18].values.flatten().astype(str)
    symptoms = list(set(symptoms))
    symptoms.sort()
    with open(file_symp, 'wb') as filehandler:
        pickle.dump(symptoms, filehandler)


def encoder(lista):
    with open(file_symp, 'rb') as filehandler:
        sym = pickle.load(filehandler)

    sym.remove(sym[0])
    vector = np.zeros(len(sym))
    for x in lista:
        if x in sym:
            vector[sym.index(x)] = 1

    return vector


def main():

    df = DATASET.copy()

    with open(file_symp, 'rb') as filehandler:
        symptoms = pickle.load(filehandler)

    df_ohe = pd.DataFrame(np.zeros((4920, 132)))
    df_ohe = df_ohe.set_axis(symptoms, axis=1, inplace=False)
    df_ohe.insert(0, 'Disease', df['Disease'])

    #
    # # Dataset encoding
    #
    for y in range(4920):
         for x in range(1, 18):
            if df.iloc[y, x] in symptoms:
                df_ohe.loc[y, df.iloc[y, x]] = 1


    x_train, x_test, y_train, y_test = train_test_split(df_ohe.iloc[:, 2:], df_ohe['Disease'], test_size=0.4,
                                                              random_state=1000)

    print("----Training----")

    CV = 10

    print("Decision Tree")

    if os.path.exists(filename2):
        print("El modelo DT existe")

    else:
        parameters = {'criterion': ('gini', 'entropy')}
        dt = DecisionTreeClassifier(random_state=42)
        clf = GridSearchCV(dt, parameters)

        clf.fit(x_train, y_train)
        pickle.dump(clf, open(filename2, 'wb'))


    print("SGDC")

    if os.path.exists(filename3):
        print("El modelo SGD existe")

    else:
        parameters = {'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),'penalty': ('l2', 'l1', 'elasticnet'),
                      'alpha': [0.0001, 0.001, 0.01], 'epsilon': [0.1, 0.01, 0.001]}

        sgd = SGDClassifier(random_state=42)
        clf = GridSearchCV(sgd, parameters)
        clf.fit(x_train, y_train)
        pickle.dump(clf, open(filename3, 'wb'))
        y_prob = clf.score(x_test, y_test)
        print("Score = {:.3%}".format(y_prob))

    print("Random Forest")

    if os.path.exists(filename4):
        print("El modelo RF existe")

    else:
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        clf.fit(x_train, y_train)
        pickle.dump(clf, open(filename4, 'wb'))
        y_prob = clf.score(x_test, y_test)
        print("Score = {:.3%}".format(y_prob))

    print("Logistic Regression")

    if os.path.exists(filename1):
        print("El modelo LR existe")

    else:
        parameters = {'penalty': ('l2', 'l1', 'elasticnet'), 'verbose': [1, 2, 10]}
        lr = LogisticRegression(max_iter=100, n_jobs=-1, penalty='l2', verbose=1)
        clf = GridSearchCV(lr, parameters)
        clf.fit(x_train, y_train)
        pickle.dump(clf, open(filename1, 'wb'))
        y_prob = clf.score(x_test, y_test)
        print("Score = {:.3%}".format(y_prob))

    print("Final")


if __name__ == '__main__':

    main()

    vect = encoder([' cough'])

    print("Loading Model 1")
    loaded_model1 = pickle.load(open(filename1, 'rb'))
    x = loaded_model1.predict(vect.reshape(1, -1))
    print(x)

    print("------------------")

    print("Loading Model 2")
    loaded_model2 = pickle.load(open(filename2, 'rb'))
    x = loaded_model2.predict(vect.reshape(1, -1))
    print(x)

    print("------------------")

    print("Loading Model 3")
    loaded_model3 = pickle.load(open(filename3, 'rb'))
    x = loaded_model3.predict(vect.reshape(1, -1))
    print(x)

    print("------------------")

    print("Loading Model 4")
    loaded_model4 = pickle.load(open(filename4, 'rb'))
    x = loaded_model4.predict(vect.reshape(1, -1))
    print(x)



