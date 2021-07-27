#%%
import time
tic = time.time()

import numpy as np
import os
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import cv2
#from facenet_pytorch import MTCNN

from tkinter import *
from tkvideo import *
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk

#FACTORY METHODDAKİ SORUN : NESNE OLUŞTURURKEN BİRBİRİNİ BEKLEMİYOR -> "QUEUE-THREAD" ile yeniden yazılacak

#FactoryMethod
from faceLib.methodFactory import MethodFactory 
metFac = MethodFactory()
(combinedClass, combinedClassName, sure) = metFac.createMatchedClass()

#%%
#Kamerayı Ayarla
mSource = 0
cap = cv2.VideoCapture(mSource)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '5'));
cap.set(cv2.CAP_PROP_FPS, 30)

toc = time.time()
print("Deploy Süresi:~ " + str(toc-tic))
#En son Deploy:  

#%%
ekran = tk.Tk()
ekran.geometry("1280x720")
ekran.title('Uygulama')

sol = Frame(ekran, bg = "#add8e6")
sol.place(relx=0.01, rely=0.01, relwidth=0.8, relheight=0.95)

sag = Frame(ekran, bg = "#bdd8ea")
sag.place(relx=0.82, rely=0.01, relwidth=0.2, relheight=0.95)

canvas = Canvas(sol, width=1280, height=720, bg='black')
canvas.pack() #expand=YES, fill=BOTH

state = -1
liste = ['Default'] + combinedClassName

def changed(event):
    text=selected.get()
    global state
    print("text" + text)
    if text == liste[0]:
        state = -1
    else:
        indis = np.where(np.array(liste) == text)
        state = int(indis[0]) - 1
    #msg = f'You selected {cbox.get()}!'
    #showinfo(title='Seçilen', message=msg)

#Combobox
selected = tk.StringVar(sag)
cbox = ttk.Combobox(sag, textvariable=selected)
cbox['values'] = liste
cbox['state'] = 'readonly'  # normal
cbox.pack(padx=3, pady=3)
cbox.bind('<<ComboboxSelected>>', changed)

def ekranaBas(frame, canvas, ekran):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(frame),1)
    canvas.create_image(0, 0, image=imgtk, anchor=NW)
    ekran.update()
    

def run():
    
    success, frame = cap.read()
    
    if success:
        tahmin = ""
        if state == -1:
            ekranaBas(frame, canvas, ekran)
        else:
            tahmin, frame  = combinedClass[state].run2(frame)
            ekranaBas(frame, canvas, ekran)
            
    ekran.after(0, run)
run()
ekran.mainloop()
print("Mainloop")
#ekran.destroy()


#%%
cap.release()
ekran.destroy();


# %%
f = open("encodes/face_enc_VGG", "wb")
f.close()
# %%
