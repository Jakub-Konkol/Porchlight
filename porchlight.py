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
from ToolTip import ToolTip


# parameters = {}
# funcCode = 'snv'
# functions['funcCode'](**parameters)

class PreprocessSelector(tk.Frame):
    def __init__(self, parent, categories, labels, column, row):
        # magic something
        super().__init__(parent)

        # make the frame of the step
        box = tk.LabelFrame(parent, text="Preprocessing 1")
        box.grid(row=row, stick=tk.W)

        #make the category combobox
        self.category = ttk.Combobox(box, width=15, state='readonly')
        self.category['values'] = tuple(categories.keys())
        self.category.grid(column=0, row=1)
        self.category.bind("<<ComboboxSelected>>", self.update_methods)

        # make the method combobox
        self.method = ttk.Combobox(box, width=15, state='readonly')
        self.method.grid(column=1, row=1)
        self.method.bind("<<ComboboxSelected>>", self.update_labels)

        # bad juju but make an empty dictionary that we'll be appending to later
        self.opt_labels = []
        self.options = []

        for ii in range(4):
            # instantiate the label and entry box
            # place the label and option into the grid, then remove it
            # so we can call it later with the options
            label = ttk.Label(box, text='', width=12)
            option = ttk.Entry(box, width=12, state='disabled')

            label.grid(column=ii+2, row=0)
            option.grid(column=ii+2, row=1)

            label.grid_remove()
            option.grid_remove()

            self.opt_labels.append(label)
            self.options.append(option)

        # define the categories and label relations
        self.categories = {'Trim': ['Trim', 'Inverse Trim'],
                           'Baseline Correction': ['AsLS', 'Polyfit'],
                           'Smoothing': ['Rolling', 'Savitzky-Golay'],
                           'Normalization': ['SNV', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max'],
                           'Center': ['Mean', 'Last Point'],
                           'Derivative': ['SG Derivative'],
                           '': ''}

        self.labels = {"Trim": ["Start", "End"],
                       "Inverse Trim": ['Start', 'End'],
                       "AsLS": ["Penalty", "Asymmetry"],
                       "Polyfit": ["Order", "Iterations"],
                       "Rolling": ["Window"],
                       "Savitzky-Golay": ["Window", "Poly. Order"],
                       'SNV': [],
                       'MSC': ["Reference"],
                       'Area': [],
                       'Peak Normalization': ["Peak position"],
                       'Vector': [],
                       'Min-max': ['Min', 'Max'],
                       'Mean': ['Self'],
                       'Last Point': [],
                       'SG Derivative': ['Window', 'Polynomial', 'Deriv. Order'],
                       '': []}

    def update_methods(self, event):
        # this will update the method combobox once the category is selected
        self.method['values'] = tuple(self.categories[self.category.get()])
        self.method.set('')
        self.update_labels(None)

    def update_labels(self, event):
        # this will update the labels and present the option boxes
        for label, option in zip(self.opt_labels, self.options):
            label.grid_forget()
            option.grid_forget()
            option.config(state='disable')
            label.config(text='')

        for ii in range(len(self.labels[self.method.get()])):
            self.opt_labels[ii].grid(row=0, column=ii+2)
            self.opt_labels[ii].config(text=self.labels[self.method.get()][ii])
            self.options[ii].grid(row=1, column=ii+2)
            self.options[ii].config(state='enable')

    def get_pp_function(self):
        # returns the method name
        return self.method.get()

    def get_pp_params(self):
        # returns a list of the parameters passed into the
        return [float(x.get()) if x.get() != '' else None for x in self.options]

class OOP():
    def __init__(self):  # Initializer method
        # Create instance
        self.win = tk.Tk()

        self.userData = SpectralData()

        # Add a title
        self.win.title("Porchlight")

        self.categories = {'Trim': ['Trim', 'Inverse Trim'],
                           'Baseline Correction': ['AsLS', 'Polyfit'],
                           'Smoothing': ['Rolling', 'Savitzky-Golay'],
                           'Normalization': ['SNV', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max'],
                           'Center': ['Mean', 'Last Point'],
                           'Derivative': ['SG Derivative'],
                           '': ''}

        self.labels = {"Trim": ["Start", "End"],
                       "Inverse Trim": ['Start', 'End'],
                       "AsLS": ["Penalty", "Asymmetry"],
                       "Polyfit": ["Order", "Iterations"],
                       "Rolling": ["Window"],
                       "Savitzky-Golay": ["Window", "Poly. Order"],
                       'SNV': [],
                       'MSC': ["Reference"],
                       'Area': [],
                       'Peak Normalization': ["Peak position"],
                       'Vector': [],
                       'Min-max': ['Min', 'Max'],
                       'Mean': ['Self'],
                       'Last Point': [],
                       'SG Derivative': ['Window', 'Polynomial', 'Deriv. Order']}

        self.create_widgets()

    def select_files(self):
        filetypes = (
            ('CSV Files', '*.csv *.CSV'),
            ('SPC Files', '*.spc *.SPC'),
            ('Text Files', '*.txt *.TXT'),
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
                              'SG Derivative': self.userData.SGDeriv,
                              'Polyfit': self.userData.polyfit}
            # 'AsLS': self.userData.asls,
            self.perform_preprocessing(None)
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
                #yunits = '(a.u.)'

            if invert_x:
                self.axis.invert_xaxis()

            self.axis.set_xlabel(xlabel, fontweight="bold", fontsize=16)
            self.axis.set_ylabel(prefix + ' ' + ylabel + ' ' + yunits, fontweight="bold", fontsize=16)
        else:
            self.axis.set_xlabel('Energy', fontweight="bold", fontsize=16)
            self.axis.set_ylabel('Intensity (a.u.)', fontweight="bold", fontsize=16)

        #self.fig.tight_layout()

        self.canvas.draw()

    def update_methods(self, event, value):
        self.method[value].set('')
        self.method[value]['values'] = tuple(self.categories[event.widget.get()])

    def update_labels(self, event, value):
        #i know this is bad
        if len(self.labels[event.widget.get()]) == 0:
            self.labelA[value].config(text='')
            self.optA[value].config(state='disable')
            self.labelB[value].config(text='')
            self.optB[value].config(state='disable')
            self.labelC[value].config(text='')
            self.optC[value].config(state='disable')
            self.perform_preprocessing(None)

        elif len(self.labels[event.widget.get()]) == 1:
            self.labelA[value].config(text=self.labels[event.widget.get()][0])
            self.optA[value].config(state='enable')
            self.labelB[value].config(text='')
            self.optB[value].config(state='disable')
            self.labelC[value].config(text='')
            self.optC[value].config(state='disable')

        elif len(self.labels[event.widget.get()]) == 2:
            self.labelA[value].config(text=self.labels[event.widget.get()][0])
            self.optA[value].config(state='enable')
            self.labelB[value].config(text=self.labels[event.widget.get()][1])
            self.optB[value].config(state='enable')
            self.labelC[value].config(text='')
            self.optC[value].config(state='disable')

        elif len(self.labels[event.widget.get()]) == 3:
            self.labelA[value].config(text=self.labels[event.widget.get()][0])
            self.optA[value].config(state='enable')
            self.labelB[value].config(text=self.labels[event.widget.get()][1])
            self.optB[value].config(state='enable')
            self.labelC[value].config(text=self.labels[event.widget.get()][2])
            self.optC[value].config(state='enable')

    def perform_preprocessing(self, event):
        print("Performing preprocessing")
        self.userData.reset()
        for ii in range(5):
            if self.method[ii].get():
                param = [self.optA[ii].get(), self.optB[ii].get(), self.optC[ii].get()]
                param = [float(x) if x != '' else None for x in param]
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
        self.left.bind("<ButtonRelease>", lambda x: self.left.focus_set)
        self.left.grid(column=0, row=0, padx=8, pady=4)
        self.right = ttk.LabelFrame(self.win)
        self.right.grid(column=1, row=0, padx=8, pady=4, sticky=tk.E+tk.W+tk.N+tk.S)

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

        self.labelA = {}
        self.labelB = {}
        self.labelC = {}

        # This is the preprocessing rows
        for ii in range(5):
            label_0 = ttk.Label(parameter_section, text="Preprocessing " + str(ii + 1) + ": ")
            label_0.grid(column=0, row=2*ii + 1)

            category = ttk.Combobox(parameter_section, width=15, state='readonly')
            category['values'] = tuple(self.categories.keys())
            category.grid(column=1, row=2*ii + 1)
            category.bind("<<ComboboxSelected>>", lambda x, i=ii: self.update_methods(x, i))

            self.category[ii] = category

            method = ttk.Combobox(parameter_section, width=15, state='readonly')
            method.grid(column=2, row=2*ii + 1)
            method.bind("<<ComboboxSelected>>", lambda x, i=ii: self.update_labels(x, i))

            self.method[ii] = method

            opt_width = 12

            labelA = ttk.Label(parameter_section, text="", width=opt_width)
            labelA.grid(column=3, row=2*ii)
            labelB = ttk.Label(parameter_section, text="", width=opt_width)
            labelB.grid(column=4, row=2*ii)
            labelC = ttk.Label(parameter_section, text="", width=opt_width)
            labelC.grid(column=5, row=2 * ii)

            optA = ttk.Entry(parameter_section, width=opt_width, state='disable')
            optA.grid(column=3, row=2*ii + 1)
            optA.bind("<FocusOut>", self.perform_preprocessing)

            optB = ttk.Entry(parameter_section, width=opt_width, state='disable')
            optB.grid(column=4, row=2*ii + 1)
            optB.bind("<FocusOut>", self.perform_preprocessing)

            optC = ttk.Entry(parameter_section, width=opt_width, state='disable')
            optC.grid(column=5, row=2*ii + 1)
            optC.bind("<FocusOut>", self.perform_preprocessing)

            self.optA[ii] = optA
            self.optB[ii] = optB
            self.optC[ii] = optC

            self.labelA[ii] = labelA
            self.labelB[ii] = labelB
            self.labelC[ii] = labelC

        #test = PreprocessSelector(parameter_section, self.categories, self.labels, 0, 11)
        #test2 = PreprocessSelector(parameter_section, self.categories, self.labels, 0, 12)

        self.right_label = ttk.Label(self.right, text='Data Preview')
        self.right_label.grid(column=0, row=0)

        # generate the figure
        self.fig = Figure(figsize=(6, 4), constrained_layout=True)
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
        file_menu.add_command(label="Load", command=self.select_files)
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


