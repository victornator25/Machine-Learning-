from tkinter import *
import cv2 #para imagenes
from tkinter import messagebox
import matplotlib.pyplot as plt

fondo = cv2.imread('ANN_Fondo_20_80.png')
#im_resized1 = cv2.resize(im1, (350, 500), interpolation=cv2.INTER_LINEAR)

im1 = cv2.imread('ANN_sigmoid_u.png')
im_resized1 = cv2.resize(im1, (300, 300), interpolation=cv2.INTER_LINEAR)

im2 = cv2.imread('ANN_sigmoid_n.png')
im_resized2 = cv2.resize(im2, (300, 300), interpolation=cv2.INTER_LINEAR)

im3 = cv2.imread('ANN_tanh_u.png')
im_resized3 = cv2.resize(im2, (300, 300), interpolation=cv2.INTER_LINEAR)

im4 = cv2.imread('ANN_tanh_n.png')
im_resized4 = cv2.resize(im4, (300, 300), interpolation=cv2.INTER_LINEAR)



# Definimos las funciones al ejecutar al clic el botón
def clic1():
    plt.imshow(cv2.cvtColor(im_resized1, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.axis(False)
    plt.title("Sigmoid with uniform distribution")
    
def clic2():
    plt.imshow(cv2.cvtColor(im_resized2, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.axis(False)
    plt.title("Sigmoid with normal distribution")

def clic3():
    plt.imshow(cv2.cvtColor(im_resized3, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.axis(False)
    plt.title("tanh with uniform distribution")

def clic4():
    plt.imshow(cv2.cvtColor(im_resized4, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.axis(False)
    plt.title("tanh with normal distribution")
    

root = Tk()
root.title("Artificial Neural Networks Images")
#root.geometry('350x350')

C = Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file = "ANN_fondo_20_80.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

C.pack()

"""
root.geometry("900x900+0+0")
fondo=PhotoImage(file="ANN_Fondo_20_80.png")
lblFondo=Label(root,image=fondo).place(x=150,y=150)
"""

# Enlezamos la función a la acción del botón
boton1=Button(root, text="ANN sigmoid uniform", command=clic1).pack()
boton2=Button(root, text="ANN sigmoid normal", command=clic2).pack()
boton3=Button(root, text="ANN tanh uniform", command=clic3).pack()
boton4=Button(root, text="ANN tanh normal", command=clic4).pack()

"""
lbl1 = Label(root, text="sigmoid function")
lbl1.grid(column=0, row=0)
"""

boton1.grid(column=0, row=2)
boton2.grid(column=0, row=4)

lbl2 = Label(root, text="hiperbolic function")
lbl2.grid(column=0, row=5)

boton3.grid(column=0, row=7)
boton4.grid(column=0, row=9)

root.mainloop() 
