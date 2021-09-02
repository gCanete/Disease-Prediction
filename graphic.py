import pickle
import tkinter as tk
from tkinter import *
from tkinter import ttk
import numpy as np
from tkinter import ttk
from ttkthemes import ThemedTk

# root = tk.Tk()
root = ThemedTk()
root.title("Disease Prediction")
root.geometry("550x150")
root.resizable(False, False)
# s = ttk.Style(root)
# s.theme_names()
# s.theme_use('yaru')
root.set_theme('clearlooks')

vect = []
filename1 = "lr_model.sav"
filename2 = "dt_model.sav"
filename3 = "sgd_model.sav"
filename4 = "rf_model.sav"

loaded_model1 = pickle.load(open(filename1, 'rb'))


# Change the label text
def add(self):
    vect.append(clicked.get())
    label2.config(text=vect)


def predict():
    symptoms = encoder(vect)
    x = loaded_model1.predict(symptoms.reshape(1, -1))
    label2.config(text=x)


def clear():
    vect.clear()
    label2.config(text=vect)


def encoder(lista):
    with open("symptoms.p", 'rb') as filehandler:
        sym = pickle.load(filehandler)

    sym.remove(sym[0])
    vector = np.zeros(len(sym))
    for x in lista:
        if x in sym:
            vector[sym.index(x)] = 1

    return vector


# Dropdown menu options
with open("symptoms.p", 'rb') as filehandler:
        options = pickle.load(filehandler)

# datatype of menu text
clicked = StringVar()

# initial menu text
clicked.set("")

# Create Dropdown menu
label1 = ttk.Label(root, text="Select your symptoms:")
label1.grid(row=0, column=0)

drop = ttk.OptionMenu(root, clicked, *options, command=add)
drop.config(width=50)
drop.grid(row=0, column=1)

# Create button, it will change label text
# button_add = Button(root, text="Add Symptom", command=add, padx=50, pady=20, bg="#d8e2dc")
button_predict = ttk.Button(root, text="Predict", command=predict)
button_clear = ttk.Button(root, text="Clear", command=clear)

# Create Label
label2 = ttk.Label(root, text=" ")
label2.grid(row=3, columnspan=3)


#Visual
# button_add.grid(row=1, column=0)
button_predict.grid(row=1, column=0, padx=50, pady=20,  sticky='w')
button_clear.grid(row=1, column=1, padx=50, pady=20,  sticky='w')

root.mainloop()