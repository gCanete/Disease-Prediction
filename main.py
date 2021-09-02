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
# symptoms = df.iloc[:, 1:18].values.flatten()
# symptoms = list(set(symptoms))
# with open("symptoms.p", 'wb') as filehandler:
#     pickle.dump(symptoms, filehandler)


def encoder(lista):
    with open("symptoms.p", 'rb') as filehandler:
        sym = pickle.load(filehandler)

    sym.remove(sym[0])
    vector = np.zeros(len(sym))
    for x in lista:
        if x in sym:
            vector[sym.index(x)] = 1

    return vector


def main():
    df = DATASET.copy()
    df_sev = SEVERITY.copy()

    with open("symptoms.p", 'rb') as filehandler:
        symptoms = pickle.load(filehandler)

    #
    # Creating a list with all the symptoms and diseases
    #

    # disease_list = []
    # symptoms_list = []
    # print(df.shape)
    #
    # disease_list = df['Disease']
    # disease_list = list(set(disease_list))
    #
    # symptoms_list = df.iloc[:, 1:18].values.flatten()
    # symptoms_list = list(set(symptoms_list))
    # symptoms_list[0] = 'Disease'
    #


    df_ohe = pd.DataFrame(np.zeros((4920, 132)))
    df_ohe = df_ohe.set_axis(symptoms, axis=1, inplace=False)
    df_ohe.insert(0, 'Disease', df['Disease'])
    # df_ohe['Disease'] = df['Disease'].factorize()[0]

    #
    # # Dataset encoding
    #
    for y in range(4920):
         for x in range(1, 18):
            if df.iloc[y, x] in symptoms:
                df_ohe.loc[y, df.iloc[y, x]] = 1

    # df_ohe.drop([0], axis=1)
    df_ohe.to_pickle("encoded_df.pkl")

    df_ohe = pd.read_pickle("encoded_df.pkl")

    x_train, x_test, y_train, y_test = train_test_split(df_ohe.iloc[:, 2:], df_ohe['Disease'], test_size=0.4,
                                                              random_state=1000)

    print("----Training----")

    CV = 10

    print("Decision Tree")
    parameters = {'criterion': ('gini', 'entropy')}
    dt = DecisionTreeClassifier(random_state=42)
    # clf = GridSearchCV(dt, parameters)

    dt.fit(x_train, y_train)
    filename = "dt_model.sav"
    pickle.dump(dt, open(filename, 'wb'))
    y_prob = dt.score(x_test, y_test)
    print("Score = {:.3%}".format(y_prob))

    clf = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(clf, df_ohe.iloc[:, 1:], df['Disease'], cv=CV)
    print(scores)

    print("SGDC")
    # parameters = {'loss': ( 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),'penalty': ('l2', 'l1', 'elasticnet'),
    #               'alpha': [0.0001, 0.001, 0.01], 'epsilon': [0.1, 0.01, 0.001]}

    parameters = {'alpha': [0.01], 'epsilon': [0.1], 'loss': 'log', 'penalty': 'l2'}
    sgd = SGDClassifier(random_state=42, n_jobs=-1, alpha=0.01, epsilon=0.1, loss='log', penalty='l2')
    # clf = GridSearchCV(sgd, parameters)
    sgd.fit(x_train, y_train)
    filename = "sgd_model.sav"
    pickle.dump(sgd, open(filename, 'wb'))
    y_prob = sgd.score(x_test, y_test)
    print("Score = {:.3%}".format(y_prob))
    # print(clf.best_params_)

    sgd = SGDClassifier(random_state=42, n_jobs=-1, alpha=0.01, epsilon=0.1, loss='log', penalty='l2')
    scores = cross_val_score(clf, df_ohe.iloc[:, 1:], df['Disease'], cv=CV)
    print(scores)


    print("Random Forest")
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train)
    filename = "rf_model.sav"
    pickle.dump(clf, open(filename, 'wb'))
    y_prob = clf.score(x_test, y_test)
    print("Score = {:.3%}".format(y_prob))

    # clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    # scores = cross_val_score(clf, df.iloc[:, 1:18], df['Disease'], cv=CV)
    # print(scores)

    print("Logistic Regression")
    parameters = {'penalty': ('l2', 'l1', 'elasticnet'), 'verbose': [1, 2, 10]}
    lr = LogisticRegression(max_iter=100, n_jobs=-1, penalty='l2', verbose=1)
    # clf = GridSearchCV(lr, parameters)
    lr.fit(x_train, y_train)
    filename = "lr_model.sav"
    pickle.dump(lr, open(filename, 'wb'))
    y_prob = lr.score(x_test, y_test)
    print("Score = {:.3%}".format(y_prob))
    mu_prob = lr.predict_proba(x_test[:100])

    # lr = LogisticRegression(max_iter=100, n_jobs=-1, penalty='l2', verbose=1)
    # scores = cross_val_score(lr, df.iloc[:, 1:18], df['Disease'], cv=CV)
    # print(scores)

    print("Final")


if __name__ == '__main__':

    main()
    filename1 = "lr_model.sav"
    filename2 = "dt_model.sav"
    filename3 = "sgd_model.sav"
    filename4 = "rf_model.sav"

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



