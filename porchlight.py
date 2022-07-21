"""
Created on Sat Apr  9 15:17:03 2022

@author: Jakub
"""
# ===========================
# Imports
# ===========================

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import messagebox as msg
from tkinter import filedialog as fd
from tkinter import Spinbox
from time import sleep  # careful - this can freeze the GUI
import pathlib

from functools import partial

from spectralData import SpectralData
#from ToolTip import ToolTip


# parameters = {}
# funcCode = 'snv'
# functions['funcCode'](**parameters)

class PreprocessSelector(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.category_val = tk.StringVar()
        self.method_val = tk.StringVar()
        self.category = ttk.Combobox()

        self.method = ttk.Combobox(left, width=12, textvariable=method_val)
        self.method['values'] = ('Trim', 'Normalize', 'Smooth')
        self.method.grid(column=0, row=0)

        self.category = ttk.Combobox(left, width=12, textvariable=method_val)
        self.category['values'] = ('Trim', 'Normalize', 'Smooth')
        self.category.grid(column=1, row=0)


class OOP():
    def __init__(self):  # Initializer method
        # Create instance
        self.win = tk.Tk()

        self.userData = SpectralData()

        # Add a title
        self.win.title("Porchlight")

        self.categories = {'Trim': ['Trim', 'Inverse Trim'],
                           # 'Baseline Correction': ['AsLS'],
                           'Smoothing': ['Rolling', 'Savitzky-Golay'],
                           'Normalization': ['SNV', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max'],
                           'Center': ['Mean', 'Last Point'],
                           'Derivative': ['SG Derivative'],
                           '': ''}

        self.create_widgets()

    def select_files(self):
        filetypes = (
            ('CSV Files', '*.csv'),
            ('SPC Files', '*.spc'),
            ('Text Files', '*.txt'),
            ('All Files', '*.*'))

        filenames = fd.askopenfilenames(
            title='Select spectra files',
            initialdir='/',
            filetypes=filetypes)

        # Check if files were selected and then load them into SpectralData
        if not filenames:
            print('No files selected')
        else:
            self.userData = SpectralData(filenames)
            self.functions = {'reset': self.userData.reset,
                              'Trim': self.userData.trim,
                              'Inverse Trim': self.userData.invtrim,
                              'Rolling': self.userData.rolling,
                              'Savitzky-Golay': self.userData.SGSmooth,
                              'SNV': self.userData.snv,
                              'MSC': self.userData.msc,
                              'Area': self.userData.area,
                              'Peak Normalization': self.userData.peaknorm,
                              'Vector': self.userData.vector,
                              'Min-max': self.userData.minmax,
                              'Mean': self.userData.mean_center,
                              'Last Point': self.userData.lastpoint,
                              'SG Derivative': self.userData.SGDeriv}
            # 'AsLS': self.userData.asls,
            self.plot_data()

    def plot_data(self):
        self.axis.clear()
        self.axis.plot(self.userData.spc.transpose())

        # make the tick labels bold
        labels = self.axis.get_xticklabels() + self.axis.get_yticklabels()
        for label in labels:
            label.set_fontweight('bold')

        spec_type = self.spectroscopy_type.get()

        prefix = ''
        invert_x = False

        if spec_type:
            if spec_type == 'IR - Transmission':
                xlabel = 'Wavenumber (cm$^{-1}$)'
                ylabel = 'Transmittance'
                yunits = '(%)'
                invert_x = True
            elif spec_type == 'IR - Absorbance':
                xlabel = 'Wavenumber (cm$^{-1}$)'
                ylabel = 'Absorbance'
                yunits = ''
                invert_x = True
            elif spec_type == 'Raman':
                xlabel = 'Raman Shift (cm$^{-1}$)'
                ylabel = 'Intensity'
                yunits = '(counts)'
            elif spec_type == 'UV-Vis':
                xlabel = 'Wavelength, (nm)'
                ylabel = 'Absorbance'
                yunits = ''

            norm_check = []
            for ii in range(5):
                if self.category[ii].get():
                    norm_check.append(self.category[ii].get())

            if 'Normalization' in norm_check or 'Center' in norm_check:
                prefix = 'Normalized'
                yunits = '(a.u.)'

            if invert_x:
                self.axis.invert_xaxis()

            self.axis.set_xlabel(xlabel, fontweight="bold", fontsize=16)
            self.axis.set_ylabel(prefix + ' ' + ylabel + ' ' + yunits, fontweight="bold", fontsize=16)
        else:
            self.axis.set_xlabel('Energy', fontweight="bold", fontsize=16)
            self.axis.set_ylabel('Intensity (a.u.)', fontweight="bold", fontsize=16)

        self.fig.tight_layout()

        self.canvas.draw()

    def update_methods(self, event, value):
        self.method[value].set('')
        self.method[value]['values'] = tuple(self.categories[event.widget.get()])

    def perform_preprocessing(self, event):
        self.userData.reset()
        for ii in range(5):
            if self.method[ii].get():
                if bool(self.optA[ii].get()) & bool(self.optB[ii].get()) & bool(self.optC[ii].get()):
                    param = (float(self.optA[ii].get()), float(self.optB[ii].get()), float(self.optC[ii].get()))
                elif bool(self.optA[ii].get()) & bool(self.optB[ii].get()):
                    param = (float(self.optA[ii].get()), float(self.optB[ii].get()))
                elif bool(self.optA[ii].get()):
                    param = (float(self.optA[ii].get()),)
                else:
                    param = ()

                self.functions[self.method[ii].get()](*param)
        self.plot_data()

    # update progressbar in callback loop
    def run_progressbar(self):
        self.progress_bar["maximum"] = 100
        for i in range(101):
            sleep(0.05)
            self.progress_bar["value"] = i  # increment progressbar
            self.progress_bar.update()  # have to call update() in loop
        self.progress_bar["value"] = 0  # reset/clear progressbar

    def start_progressbar(self):
        self.progress_bar.start()

    def stop_progressbar(self):
        self.progress_bar.stop()

    def progressbar_stop_after(self, wait_ms=1000):
        self.win.after(wait_ms, self.progress_bar.stop)

    def export_data(self):
        ext_options = (("csv", ".csv"),
                       ("Excel", ".xlsx"),
                       ("All Files", "."))
        f_path = fd.asksaveasfilename(filetypes=ext_options, defaultextension=".csv")
        if f_path is None:
            return
        elif pathlib.Path(f_path).suffix == ".csv":
            self.userData.spc.to_csv(f_path)
        elif pathlib.Path(f_path).suffix == ".xlsx":
            self.userData.spc.to_excel(f_path)

    # Exit GUI cleanly
    def _quit(self):
        self.win.quit()
        self.win.destroy()
        exit()

        #####################################################################################

    def create_widgets(self):

        # set up the left/right frames
        self.left = ttk.Frame(self.win)
        self.left.grid(column=0, row=0, padx=8, pady=4)
        self.right = ttk.LabelFrame(self.win)
        self.right.grid(column=1, row=0, padx=8, pady=4)

        # set up the three panels of left
        setup_section = ttk.LabelFrame(self.left, text='Spectroscopy Type and Load')
        parameter_section = ttk.LabelFrame(self.left, text='Parameter Selection')
        export_section = ttk.LabelFrame(self.left, text='Export')

        setup_section.grid(column=0, row=0)
        parameter_section.grid(column=0, row=1)
        export_section.grid(column=0, row=2)

        # making the spectroscopy type combobox
        self.spectroscopy_type = ttk.Combobox(setup_section, width=15, state='readonly')
        self.spectroscopy_type['values'] = ('IR - Transmission', 'IR - Absorbance', 'Raman', 'UV-Vis')
        self.spectroscopy_type.grid(column=0, row=0)

        # making the load files button
        open_button = ttk.Button(
            setup_section,
            text='Load Files',
            command=self.select_files)

        open_button.grid(column=0, row=1)

        # Prepare the variables for the preprocessing selectors
        self.category_select = [tk.StringVar()] * 5
        self.method_select = [tk.StringVar()] * 5
        self.optA_val = [tk.StringVar()] * 5
        self.optB_val = [tk.StringVar()] * 5
        self.optC_val = [tk.StringVar()] * 5

        self.optA = {}
        self.optB = {}
        self.optC = {}
        self.category = {}
        self.method = {}

        # This is the preprocessing rows
        for ii in range(5):
            label_0 = ttk.Label(parameter_section, text="Preprocessing " + str(ii + 1) + ": ")
            label_0.grid(column=0, row=ii + 1)

            category = ttk.Combobox(parameter_section, width=15, state='readonly')
            category['values'] = tuple(self.categories.keys())
            category.grid(column=1, row=ii + 1)
            category.bind("<<ComboboxSelected>>", lambda x, i=ii: self.update_methods(x, i))

            self.category[ii] = category

            method = ttk.Combobox(parameter_section, width=15, state='readonly')
            method.grid(column=2, row=ii + 1)
            method.bind("<<ComboboxSelected>>", lambda x: self.perform_preprocessing(x))

            self.method[ii] = method

            optA = ttk.Entry(parameter_section, width=5)
            optA.grid(column=3, row=ii + 1)
            optA.bind("<FocusOut>", self.perform_preprocessing)

            optB = ttk.Entry(parameter_section, width=5)
            optB.grid(column=4, row=ii + 1)
            optB.bind("<FocusOut>", self.perform_preprocessing)

            optC = ttk.Entry(parameter_section, width=5)
            optC.grid(column=5, row=ii + 1)
            optC.bind("<FocusOut>", self.perform_preprocessing)

            self.optA[ii] = optA
            self.optB[ii] = optB
            self.optC[ii] = optC

        self.right_label = ttk.Label(self.right, text='Data Preview')
        self.right_label.grid(column=0, row=0)

        # generate the figure
        self.fig = Figure(figsize=(6, 4))
        self.axis = self.fig.add_subplot(111)

        self.axis.tick_params(axis="y", direction="in", labelsize=14)
        self.axis.tick_params(axis="x", direction="in", labelsize=14)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right, pack_toolbar=False)
        self.toolbar.update()
        self.canvas._tkcanvas.grid(column=0, row=1)
        self.toolbar.grid(column=0, row=2, sticky=tk.W)
        # Creating a Menu Bar
        menu_bar = Menu(self.win)
        self.win.config(menu=menu_bar)

        # Add menu items
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New")
        file_menu.add_command(label="Save", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Display a Message Box
        def _msgBox():
            msg.showinfo('Python Message Info Box', 'A Python GUI created using tkinter:\nThe year is 2022.')

            # Add another Menu to the Menu Bar and an item

        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=_msgBox)  # display messagebox when clicked
        menu_bar.add_cascade(label="Help", menu=help_menu)

        # Change the main windows icon
        self.win.iconbitmap('porchlight.ico')


# ======================
# Start GUI
# ======================
oop = OOP()
oop.win.mainloop()


