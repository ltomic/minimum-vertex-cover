import os
import sys
import tkinter as tk
import tkinter.font as tkf
import tkinter.ttk as tkk
import MVC

folder = 'datasets'
filelist = sorted([fname for fname in os.listdir(folder)], key = lambda name: name.lower())
font_family = 'Helvetica'

top = tk.Tk(className = 'Minimum vertex cover')

f1 = tkf.Font(family = font_family, size = 20, weight = 'bold')
f2 = tkf.Font(family = font_family, size = 15)

generate_weights = tk.IntVar()

w = tk.Checkbutton(top, text = "Generate weights",
                   font = f2, pady = 10, variable = generate_weights)
lp = tk.Label(top, text = "Population:", font = f2, pady = 10)
lg = tk.Label(top, text = "Generations:", font = f2, pady = 10)
lt = tk.Label(top, text = "Time limit (seconds)", font = f2, pady = 10)
p = tk.Entry(top, exportselection = 0, justify = tk.CENTER)
g = tk.Entry(top, exportselection = 0, justify = tk.CENTER)
t = tk.Entry(top, exportselection = 0, justify = tk.CENTER)
lm = tk.Label(top, text = 'Choose a graph', font = f2)
optmenu = tkk.Combobox(top, values=filelist, state='readonly', font=f2)
top.option_add("*TCombobox*Listbox*Font", f2)
b = tk.Button(top, text = "Begin!", font = f1,
              height = 2, width = 20, bg = 'black', fg = 'white',
              command = lambda: MVC.main("datasets/" + optmenu.get(), int(p.get()),
                                     int(g.get()) if len(g.get()) else float('inf'),
                                     generate_weights.get(),
                                     int(t.get()) if len(t.get()) else float('inf')))

exit_b = tk.Button(top, text = "Exit", font = f2, height = 1, width = 10,
                   command = top.destroy)

lm.pack()
optmenu.pack(fill='x')
lp.pack()
p.pack()
lg.pack()
g.pack()
lt.pack()
t.pack()
w.pack()
b.pack(pady = 30)
exit_b.pack()
top.geometry('400x500')
top.mainloop()

