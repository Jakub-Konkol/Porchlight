"""
Created on Sat Apr  9 15:17:03 2022

@author: Jakub
"""
import base64

# ===========================
# Imports
# ===========================

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from tkinter import Menu
from tkinter import messagebox as msg
from tkinter import filedialog as fd
from time import sleep  # careful - this can freeze the GUI
import pathlib
from .spectralData import SpectralData
# from ToolTip import ToolTip


class PreprocessSelector(tk.Frame):
    def __init__(self, parent, column, row, title='Step'):
        # magic something
        super().__init__(parent)

        # make the frame of the step
        box = tk.LabelFrame(parent, text=title)
        box.grid(column=column, row=row, padx=3, pady=3, stick=tk.W)

        # define the categories and label relations
        self.categories = {'Trim': ['Trim', 'Inverse Trim'],
                           'Baseline Correction': ['AsLS', 'Polyfit', 'Pearson'],
                           'Smoothing': ['Rolling', 'Savitzky-Golay'],
                           'Normalization': ['SNV', 'Detrend', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max', 'Pareto'],
                           'Center': ['Mean', 'Last Point'],
                           'Derivative': ['SG Derivative'],
                           'Dataset': ['Subtract', 'Reset'],
                           '': ''}

        self.labels = {"Trim": ["Start", "End"],
                       "Inverse Trim": ['Start', 'End'],
                       "AsLS": ["Penalty", "Asymmetry", "Iterations"],
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
                       'Reset': [],
                       'Subtract': ['Spectrum'],
                       'Detrend': ['Order'],
                       'Pareto': [],
                       'Pearson': ['u', 'v'],
                       '': []}

        # make the category combobox
        self.category = ttk.Combobox(box, width=17, state='readonly')
        self.category['values'] = tuple(self.categories.keys())
        self.category.grid(column=0, row=1, padx=3, pady=3)
        self.category.bind("<<ComboboxSelected>>", self.update_methods)
        ttk.Label(box, text='Category', width=17).grid(column=0, row=0, sticky=tk.W, padx=3)

        # make the method combobox
        self.method = ttk.Combobox(box, width=17, state='readonly')
        self.method.grid(column=1, row=1)
        self.method.bind("<<ComboboxSelected>>", self.update_labels)
        ttk.Label(box, text='Technique', width=17).grid(column=1, row=0, sticky=tk.W, padx=3)

        # bad juju but make an empty dictionary that we'll be appending to later
        self.opt_labels = []
        self.options = []

        for ii in range(4):
            # instantiate the label and entry box place the label and option into the grid, then remove it
            # so we can call it later with the options
            label = ttk.Label(box, text='', width=12)
            option = ttk.Entry(box, width=12, state='disabled')

            label.grid(column=ii + 2, row=0, padx=2)
            option.grid(column=ii + 2, row=1, padx=2)

            label.grid_remove()
            option.grid_remove()

            self.opt_labels.append(label)
            self.options.append(option)

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
            self.opt_labels[ii].grid(row=0, column=ii + 2)
            self.opt_labels[ii].config(text=self.labels[self.method.get()][ii])
            self.options[ii].grid(row=1, column=ii + 2)
            self.options[ii].config(state='enable')

    def get_pp_function(self):
        # returns the method name
        return self.method.get()

    def get_pp_params(self):
        # returns a list of the parameters passed into the entry
        return [float(x.get()) if x.get() != '' else None for x in self.options]

    def get_pp_category(self):
        # returns the category of the technique
        return self.category.get()


class OOP:
    def __init__(self):  # Initializer method
        # Create instance
        self.win = tk.Tk()

        # i still like Breeze but the application becomes less responsive, artifacting as it resizes
        # self.win.tk.call("source", "Breeze.tcl")
        # ttk.Style().theme_use("Breeze")

        self.userData = SpectralData()

        # Add a title
        self.win.title("Porchlight")

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
                              'Polyfit': self.userData.polyfit,
                              'AsLS': self.userData.AsLS,
                              'Reset': self.userData.reset,
                              'Subtract': self.userData.subtract,
                              'Detrend': self.userData.detrend,
                              'Pareto': self.userData.pareto,
                              'Pearson': self.userData.pearson}
            self.perform_preprocessing()
            self.plot_data()

    def plot_data(self):
        self.axis.clear()
        self.axis.plot(self.userData.spc.transpose())
        if self.userData.spc.shape[0] < 10:
            self.axis.legend([str(x+1) for x in range(self.userData.spc.shape[0])], loc='best', ncol=1)
        elif self.userData.spc.shape[0] < 20:
            self.axis.legend([str(x+1) for x in range(self.userData.spc.shape[0])], loc='best', ncol=2)


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
            for step in self.pp_steps:
                if step.get_pp_function():
                    norm_check.append(step.get_pp_category())

            if 'Normalization' in norm_check or 'Center' in norm_check:
                prefix = 'Normalized'
                yunits = ''

            if invert_x:
                self.axis.invert_xaxis()

            self.axis.set_xlabel(xlabel, fontweight="bold", fontsize=16)
            self.axis.set_ylabel(prefix + ' ' + ylabel + ' ' + yunits, fontweight="bold", fontsize=16)
        else:
            self.axis.set_xlabel('Energy', fontweight="bold", fontsize=16)
            self.axis.set_ylabel('Intensity (a.u.)', fontweight="bold", fontsize=16)

        # self.fig.tight_layout()

        self.canvas.draw()

    def perform_preprocessing(self):
        self.userData.reset()
        for step in self.pp_steps:
            if step.get_pp_function():
                param = step.get_pp_params()
                self.functions[step.get_pp_function()](*param)
        self.plot_data()

    def add_step(self):
        pp_step = PreprocessSelector(self.parameter_section, 0, len(self.pp_steps),
                                     'Step' + str(len(self.pp_steps) + 1))
        self.pp_steps.append(pp_step)

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
            self.userData.export_csv(f_path)
        elif pathlib.Path(f_path).suffix == ".xlsx":
            self.userData.export_excel(f_path)

    def reset_focus(self):
        # why doesn't this work
        self.win.focus_set()

    # Exit GUI cleanly
    def _quit(self):
        self.win.quit()
        self.win.destroy()
        exit()

        #####################################################################################

    def create_widgets(self):
        import os

        # set up the left/right frames
        self.left = ttk.Frame(self.win)
        self.left.bind("<ButtonRelease>", lambda x: self.reset_focus)
        self.left.grid(column=0, row=0, padx=8, pady=4)
        self.right = ttk.LabelFrame(self.win)
        self.right.grid(column=1, row=0, padx=8, pady=4, sticky=tk.E + tk.W + tk.N + tk.S)

        # set up the three panels of left
        setup_section = ttk.LabelFrame(self.left, text='Spectra type and load')
        self.parameter_section = ttk.LabelFrame(self.left, text='Parameter selection')
        export_section = ttk.LabelFrame(self.left, text='Save and export')

        setup_section.grid(column=0, row=0)
        self.parameter_section.grid(column=0, row=1)
        self.parameter_section.bind("<<Button-1>>", self.reset_focus)
        export_section.grid(column=0, row=4)

        # making the spectroscopy type combobox
        self.spectroscopy_type = ttk.Combobox(setup_section, width=15, state='readonly')
        self.spectroscopy_type['values'] = ('IR - Transmission', 'IR - Absorbance', 'Raman', 'UV-Vis')
        self.spectroscopy_type.grid(column=0, row=0, padx=3, pady=3)

        # making the load files button
        open_button = ttk.Button(
            setup_section,
            text='Load Files',
            command=self.select_files)

        open_button.grid(column=0, row=1, padx=3, pady=3)

        self.pp_steps = []

        for ii in range(5):
            pp_step = PreprocessSelector(self.parameter_section, 0, ii, "Step " + str(ii + 1))
            self.pp_steps.append(pp_step)

        add_step_button = ttk.Button(
            self.left,
            text="Add Step",
            command=self.add_step
        )
        add_step_button.grid(column=0, row=2, padx=2, pady=2)

        pp_button = ttk.Button(
            self.left,
            text='Apply Preprocessing',
            command=self.perform_preprocessing)

        pp_button.grid(column=0, row=3, padx=2, pady=2)

        save_button = ttk.Button(
            self.left,
            text="Save data",
            command=self.export_data
        )

        save_button.grid(column=0, row=4, padx=2, pady=2)

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
            msg.showinfo('About Porchlight', 'Porchlight was created by Jakub Konkol and George Tsilomelekis.\nCheck us out at gtsilomelekis.com!\n\nThis version: 1.1.1')

        # Add another Menu to the Menu Bar and an item
        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=_msgBox)  # display messagebox when clicked
        menu_bar.add_cascade(label="Help", menu=help_menu)

        # # Change the main windows icon, using ico if windows, otherwise png
        resource_dir = os.path.join(os.path.dirname(__file__), 'resources')
        if os.name == 'nt':
            import ctypes
            self.win.iconbitmap(os.path.join(resource_dir, 'porchlight.ico'))
            myappid = 'Rutgers.Porchlight.1.1.1'  # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        else:
            # weird linux hack from https://stackoverflow.com/questions/45361749/python-3-6-tkinter-window-icon-on-linux-error,
            # not sure it works.
            with open(os.path.join(resource_dir, 'porchlight.gif'), 'rb') as icon_gif:
                icon_base64 = base64.b64encode(icon_gif.read())
            self.win.iconbitmap(icon_base64)


if __name__ == "__main__":
    # ======================
    # Start GUI
    # ======================
    oop = OOP()
    oop.win.mainloop()
