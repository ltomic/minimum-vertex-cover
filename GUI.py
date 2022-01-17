import os
import sys
import tkinter as tk
import tkinter.font as tkf
import tkinter.ttk as tkk
import minimumVertexCover
import runpy

folder = 'datasets'
filelist = [fname for fname in os.listdir(folder)]

top = tk.Tk(className = 'Minimum vertex cover')

f = tkf.Font(family='Helvetica', size=20, weight='bold')

random_weights = tk.IntVar()
filename = tk.StringVar()
population_size = tk.IntVar()
n_gen = tk.IntVar()

w = tk.Checkbutton(top, text = "Generate random weights",
                   font = '20', pady = 50, variable = random_weights)
lp = tk.Label(top, text = "Population:", font = '20',
              pady = 15)
lg = tk.Label(top, text = "Generations:", font = '20',
              pady = 15)
p = tk.Entry(top, exportselection = 0, justify = tk.CENTER)
g = tk.Entry(top, exportselection = 0, justify = tk.CENTER)
lm = tk.Label(top, text = 'Choose a graph', font = '20')
optmenu = tkk.Combobox(top, values=filelist,
                       state='readonly')
b = tk.Button(top, text = "Begin!", font = f,
              height = 20, width = 20, bg = 'black', fg = 'white',
              command = lambda: runpy.run_path("minimumVertexCover.py"))

lm.pack()
optmenu.pack(fill='x')
lp.pack()
p.pack()
lg.pack()
g.pack()
w.pack()
b.pack(side = tk.BOTTOM)
top.geometry('400x400')
top.mainloop()

