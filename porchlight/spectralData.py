# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:23:25 2022

@author: Jakub
"""


class SpectralData():
    """
    Class that handles the loading, storage and manipulation of spectral data.
    """
    def __init__(self, file=None):
        """
        Initialization of the class. Expects a list of files to load in, data handling is automatic.

        :param file: str, list-like
            Directories of files to be loaded
            If None, initializes an empty class you can populate yourself.
        """
        import numpy as np
        import pandas as pd

        if file is None:
            # empty initializer if you want to populate the values yourself.
            self._wav_raw = [0]
            self._spc_raw = [0]
            self.war = [0]
            self.spc = [0]
            self.wav = [0]
            self._baselines = []
            self.shape = [0]
            self.tf_history = []

        else:
            # initialize temporary arrays
            store_spc = []
            self.file_source = []
            self.tf_history = []

            # start with first file
            if isinstance(file, str):
                spc = self.getDataFromFile(file)
            else:
                spc = self.getDataFromFile(file[0])

            store_spc.append(spc)

            if not isinstance(file, str) and len(file) > 1:
                for ii in range(1, len(file)):
                    spc = self.getDataFromFile(file[ii])
                    store_spc.append(spc)

            combined = pd.concat(store_spc, axis=0, ignore_index=True)
            combined.sort_index(axis=1, inplace=True)

            # This section tries to rectify abscissa mismatches due to rounding errors. If two adjacent columns have
            # opposite missing values, then it will combine those columns into one with the mean abscissa value
            for ii in range(1, len(combined.columns) - 1):
                before = combined.columns[ii] - combined.columns[ii - 1]
                after = combined.columns[ii + 1] - combined.columns[ii]

                if before < after:
                    use_1 = ii - 1
                    use_2 = ii
                else:
                    use_1 = ii
                    use_2 = ii + 1

                comparison = combined.iloc[:, use_1].isna() & ~combined.iloc[:, use_2].isna() | ~combined.iloc[:,
                                                                                                 use_1].isna() & combined.iloc[
                                                                                                                 :,
                                                                                                                 use_2].isna()

                if comparison.all():
                    combined[combined.columns[use_1]] = combined.iloc[:, use_1].fillna(combined.iloc[:, use_2])
            combined.dropna(axis=1, inplace=True)

            self._spc_raw = combined.copy(deep=True)
            self.spc = self._spc_raw.copy(deep=True)
            if self._spc_raw is self.spc:
                print("True")
            self._wav_raw = self.spc.columns[:]
            self.wav = self._wav_raw.copy(deep=True)
            self.war = pd.DataFrame(np.zeros((self.spc.shape[1])))
            self.shape = self.spc.shape
            self._baselines = np.zeros(self.spc.shape)
            # self.baseline = pd.DataFrame(np.zeros(self._spc_raw.shape), columns=self._wav_raw)

    def getDataFromFile(self, file):
        """
        Perform logic to determine filetype and corresponding import function. Import data and append file to
        imported_file array to keep track.

        Parameters
        ----------
        file : STR
            Path to file.

        Returns
        -------
        spc : Pandas dataframe or series
            Array of spectral values, ordered as one spectra per column

        """
        import os
        ext = os.path.splitext(file)[-1].lower()
        # go through decision tree to get correct import function
        if ext == '.txt':
            spc = self.readTextFile(file)
        elif ext == ".spc":
            spc = self.readSPCFile(file)
        elif ext == '.jdx':
            print("J-Camp file import not implemented yet")
            # TODO: figure out j-camp import
        elif ext == '.csv':
            spc = self.readTextFile(file)
        else:
            print("This filetype is not supported")

        # this is to be able to later realign the spectra to the source file
        self.file_source = self.file_source + [os.path.split(file)[1] for x in range(spc.shape[0])]

        return spc

    def readTextFile(self, file):
        """
        Perform interpretation for txt and csv files.

        Parameters
        ----------
        file : STR
            The path to the file

        Returns
        -------
        raw : pandas series or dataframe containing the spectral axis and intensities

        """
        import pandas as pd
        import csv

        # use csv.Sniffer to determine whether there is a header
        with open(file, 'r') as f:
            first_three_lines = ''
            for ii in range(3):
                first_three_lines = first_three_lines + f.readline()
            dialect = csv.Sniffer().sniff(first_three_lines)
            has_header = csv.Sniffer().has_header(first_three_lines)
            f.close()

        if has_header:
            head_col = 0
        else:
            head_col = None

        raw = pd.read_csv(file, delimiter=dialect.delimiter, header=head_col)

        # assuming that there are more data points than spectra, make row-wise if more rows than columns
        if raw.shape[0] > raw.shape[1]:
            raw = raw.set_index(raw.columns[0]).transpose()
        else:
            raw = raw.transpose().set_index(raw.columns[0]).transpose()

        return raw

    def readCSVFile(self, file):
        """
        Perform interpretation for txt and csv files.

        Parameters
        ----------
        file : STR
            The path to the file

        Returns
        -------
        wav : NP.ARRAY
            Array of independent wavenumber values
        spc : NP.ARRAY
            Array of spectral values, ordered as one spectra per column

        """
        import numpy as np
        data = np.loadtxt(file, delimiter=',')
        # assuming that there are more data points than spectra, make columnwise
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        wav = data[:, 0]
        spc = data[:, 1:]

        # return squeezed spc in case only 1D
        return wav, np.squeeze(spc)

    def readSPCFile(self, file):
        """
        Reads .spc file using spc_spectra and returns the spectra as a pandas dataframe.
        :param file: str
            Directory of file to be loaded
        :return: pandas.DataFrame
            Spectra as a pandas Dataframe
        """
        import pandas as pd
        try:
            import spc_spectra
        except:
            print("Porchlight can use spc_spectra to import spc files.\nInstall using 'pip install spc_spectra'")
            return
        import numpy as np

        f = spc_spectra.File(file)

        # we can use the position of the '-' to figure out how to handle the x and y
        pos = f.dat_fmt.find('-')
        if pos == 0:
            # code is of type -xy, assume all have the same wav
            wav = f.sub[0].x
        elif pos == 1:
            # code is of type -xy
            wav = f.x
        elif pos == 2:
            # code is of type gx-y
            wav = f.x

        temp = []
        for ii in range(f.fnsub):
            temp.append(f.sub[ii].y)
        spc = np.array(temp)
        spc = pd.DataFrame(spc, columns=wav)
        return spc

    def readExcelFile(self, file):
        """
        Reads Excel file using pandas and returns the spectra as a pandas dataframe

        :param file: str
            Directory of file
        :return:
        """

        # TODO: figure out excel reader

        return

    def readJCAMPFile(self, file):
        """
        Reads jcamp file. Figures out if the contents is a vibrational spectrum or an NMR spectrum, then uses the
        package jcamp or nmrglue respectively.

        :param file: str
            Directory of file
        :return:

        """

        # TODO: figure out jcamp reader
        # need to do some file reading shenenigans where I look for a field called ##DATA TYPE or ##DATATYPE. Maybe re
        # where I cut out the spaces.

        return

    def readSPAFile(self, file):
        """
        Read ThermoFisher SPA files

        :param file:
        :return:
        """

        # https://github.com/lerkoah/spa-on-python maybe? no pypi tho

    def plot(self):
        self.spc.transpose().plot()

    def reset(self, *args):
        """
        Reset the spectra and wavenumber to raw imported values, undoing any preprocessing.
        """
        import pandas as pd
        import numpy as np
        self.spc = self._spc_raw.copy(deep=True)
        self.wav = self._wav_raw.copy(deep=True)
        self._baselines = pd.DataFrame(np.zeros(self._spc_raw.shape), columns=self._wav_raw)
        self.tf_history = []

    def rolling(self, window, *args):
        """
        Performs a rolling (moving window, boxcar) average of the spectra.

        :param window: int
            Number of datapoints to average together
        :param args:
            None
        :return:
            None
        """
        if window is None:
            print("Warning: no window length supplied. Defaulting to 1.")
            window = 1
        if not isinstance(window, int):
            window = int(window)

        self.tf_history.append(['rolling', {'window': window}])

        self.spc = self.spc.transpose()
        # first perform the rolling window smooth
        self.spc = self.spc.rolling(window).mean()

        # use the NANs to cut the wav and spc matrices
        self.wav = self.wav[self.spc.iloc[:, 0].notna()]
        self.spc = self.spc.dropna()

        self.spc = self.spc.transpose()

    def SGSmooth(self, window, poly, *args):
        """
        Performs Savitzky-Golay smoothing of the data as implemented by scipy.signal.

        Parameters
        ----------
        window : INT
            The window size, or number of points used in the fitting.
        poly : INT
            Order of polynomial used in the fitting.

        Returns
        -------
        None.

        """
        from scipy.signal import savgol_filter
        import pandas as pd

        if not isinstance(window, int):
            window = int(window)
        if not isinstance(poly, int):
            poly = int(poly)

        self.tf_history.append(['SGSmooth', {'window': window, 'poly': poly}])

        self.spc = pd.DataFrame(savgol_filter(self.spc, window, poly, axis=1), columns=self.spc.columns,
                                index=self.spc.index)

    def SGDeriv(self, window, poly, order, *args):
        """
        Performs Savitzky-Golay derivative of spectrum as implemented in scipy.signal

        :param window: int
            Window length, or number of datapoints to fit polynomial. Must be odd.
        :param poly: int
            Polynomial of fitting function
        :param order: int
            Order of derivative
        :param args:
            None
        :return:
            None
        """
        from scipy.signal import savgol_filter
        import pandas as pd

        self.tf_history.append(['SGDeriv', {'window': window, 'poly': poly, 'order': order}])

        if not isinstance(window, int):
            window = int(window)
        if not isinstance(poly, int):
            poly = int(poly)
        if not isinstance(order, int):
            order = int(order)

        self.spc = pd.DataFrame(savgol_filter(self.spc, window_length=window, polyorder=poly, deriv=order, axis=1),
                                columns=self.spc.columns, index=self.spc.index)

    def snv(self, *args):
        """
        Performs in-place SNV normalization of data.

        Returns
        -------
        None.

        Reference:
            Barnes, R. J., M. S. Dhanoa, and S. J. Lister, “Standard Normal Variate Transformation and De-trending of
            Near-Infrared Diffuse Reflectance Spectra,” Appl. Spectrosc., 43, pp. 772–777 (1989).

        """
        self.tf_history.append(['snv', {}])
        self.spc = self.spc.sub(self.spc.mean(axis=1), axis=0).divide(self.spc.std(axis=1), axis=0)

    def msc(self, reference=None, *args):
        """
        Performs in-place MSC scatter correction of data.

        Parameters
        ----------
        reference : INT, optional
            The value of the spectra to be fitted to. If none is provided, the average spectrum is used. The default is
            None.

        Returns
        -------
        None.

        """
        import numpy as np

        self.tf_history.append(['msc', {'reference': reference}])

        if reference is None:
            ref = self.spc.mean(axis=0)
        else:
            if isinstance(reference, int):
                ref = self.spc.iloc[reference, :]
            else:
                reference = int(reference)
                ref = self.spc.iloc[reference, :]

        for ii in range(self.spc.shape[0]):
            fit = np.polyfit(ref, self.spc.iloc[ii, :], 1, full=True)
            self.spc.iloc[ii, :] = (self.spc.iloc[ii, :] - fit[0][1]) / fit[0][0]

    def detrend(self, order=2, *args):
        """
        Performs detrending of spectrum. This does NOT perform SNV automatically as suggested by Barnes et. al., to get
        the paper version of detrending perform SNV then detrending.

        :param order: int
            Integer of polynomial to fit for detrend. Default is 2 as recommended in paper for NIR data.
        :return:
            None

        Reference:
            Barnes, R. J., M. S. Dhanoa, and S. J. Lister, “Standard Normal Variate Transformation and De-trending of
            Near-Infrared Diffuse Reflectance Spectra,” Appl. Spectrosc., 43, pp. 772–777 (1989).
        """

        import numpy as np

        self.tf_history.append(['detrend', {'order': order}])

        if order is None:
            order = 2

        for ii in range(self.spc.shape[0]):
            fit = np.polyfit(self.wav, self.spc.iloc[ii, :], order)
            self.spc.iloc[ii, :] = self.spc.iloc[ii, :].values - np.polyval(fit, self.wav)

    def trim(self, start=None, end=None, *args):
        """
        Takes a start and an end parameter for the energy axis and keeps the spectral columns that are inclusive of the
        range.

        Example: spc columns are [1, 2, 3, 4, 5]. trim(start=2, end=4) will keep only columns [2, 3, 4].

        :param start: float
            Axis value to start keeping
        :param end: float
            Axis value to end keeping
        :param args:
            None
        :return: None
        """

        self.tf_history.append(['trim', {"start": start, "end": end}])

        if start is None:
            start = self.spc.columns[0]
        if end is None:
            end = self.spc.columns[-1]

        self.spc = self.spc.transpose()
        self.spc = self.spc.loc[(self.spc.index > start) & (self.spc.index < end)]
        self.wav = self.wav[(self.wav > start) & (self.wav < end)]
        self.spc = self.spc.transpose()

    def invtrim(self, start=None, end=None, *args):
        """
        Takes a start and an end parameter for the energy axis and keeps the spectral columns that are exclusive of the
        range.

        Example: spc columns are [1, 2, 3, 4, 5]. invtrim(start=2, end=4) will keep only columns [1, 5].

        :param start: float
            Axis value to start keeping
        :param end: float
            Axis value to end keeping
        :param args:
            None
        :return: None
        """

        self.tf_history.append(['invtrim', {"start": start, "end": end}])

        if start is None:
            start = self.spc.index[0]
        if end is None:
            end = self.spc.index[-1]

        self.spc = self.spc.transpose()
        self.spc = self.spc.loc[(self.spc.index < start) | (self.spc.index > end)]
        self.wav = self.wav[(self.wav < start) | (self.wav > end)]
        self.spc = self.spc.transpose()

    def area(self, *args):
        """
        Integrates the area under the spectrum using np.trapz and then divides the spectral intensities by the area.

        :param args:
            None
        :return: None
        """
        import numpy as np
        self.tf_history.append(['area', {}])
        self.spc = self.spc.divide(self.spc.apply(lambda x: np.trapz(x, self.wav), axis=1), axis=0)

    def lastpoint(self, *args):
        """
        Subtracts each spectrum by the last point in spectrum, the intensity in the last column of spc.

        :param args:
            None
        :return:
            None
        """
        self.tf_history.append(['lastpoint', {}])
        self.spc = self.spc.sub(self.spc.iloc[:, -1], axis=0)

    def mean_center(self, option=False, *args):
        """
        Subtracts the spectra by the mean. If option==False, subtracts each spectrum by the mean of the spectrum. If
        option==True, subtracts the column by the mean of the column (energy axis).

        :param option: bool
            If False, subtracts each spectrum by the mean of the spectrum
            If True, subtracts each column by the mean of the column
        :param args:
        :return:
        """
        self.tf_history.append(['mean_center', {"option": option}])
        if not option:
            self.spc = self.spc.sub(self.spc.mean(axis=1), axis=0)
        elif option:
            self.spc = self.spc.sub(self.spc.mean(axis=0), axis=1)

    def peaknorm(self, wavenumber, *args):
        """
        Normalizes the spectrum such that the specified wavenumber is intensity is 1. If that exact wavenumber is not
        found, the closest value is selected.

        :param wavenumber: float
            Desired axis value to divide the spectrum.
        :param args:
            None
        :return:
            None
        """
        self.tf_history.append(['peaknorm', {"wavenumber": wavenumber}])
        self.spc = self.spc.transpose()
        index = self.spc.index.get_loc(wavenumber, method='nearest')
        self.spc = self.spc.divide(self.spc.iloc[index, :])
        self.spc = self.spc.transpose()

    def vector(self, *args):
        """
        Performs vector normalization of the spectrum.

        :param args:
            None
        :return:
            None
        """
        self.tf_history.append(['vector', {}])
        self.spc = self.spc.divide(((self.spc ** 2).sum(axis=1)) ** (1 / 2), axis=0)

    def minmax(self, min_val=0, max_val=1, *args):
        """
        Performs a min-max normalization of the spectrum. Unless specified, the maximum value is 1 while the minimum
        value is 0.

        :param min_val: float
            Minimum value of the resulting spectrum
        :param max_val: float
            Maximum value of the resulting spectrum
        :param args:
            None
        :return:
            None
        """
        self.tf_history.append(['minmax', {"min_val": min_val, "max_val": max_val}])
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 1

        self.spc = self.spc.transpose()
        self.spc = min_val + (self.spc.sub(self.spc.min(axis=0))) * (max_val - min_val) / (
                self.spc.max(axis=0) - self.spc.min(axis=0))
        self.spc = self.spc.transpose()

    def _get_AsLS_baseline(self, y, lam, p, niter):
        """
        This function returns the Asymmetric Least Squares baseline of a spectrum. For internal use.

        :param y: array-like
            spectrum
        :param lam: float
            Smoothness parameter, recommended to use values between 1e2 to 1e9
        :param p: float
            Asymmetry parameter, recommended to use values between 0.001 to 0.1
        :param niter: int
            maximum number of iterations
        :return:
            Fitted AsLS baseline for spectrum

        References:
            Eilers and Boelens, 2005. Baseline Correction with Asymmetric Least Squares Smoothing.
            This Python adaptation was taken from https://stackoverflow.com/a/50160920
        """
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        L = self.spc.shape[1]
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def AsLS(self, lam, p, niter=20, *args):
        """
        This function fits the Asymmetric Least Squares baseline of a spectrum and subtracts it.

        :param lam: float
            Smoothness parameter, recommended to use values between 1e2 to 1e9
        :param p: float
            Asymmetry parameter, recommended to use values between 0.001 to 0.1
        :param niter: int
            maximum number of iterations
        :return:
            None

        References:
            Eilers and Boelens, 2005. Baseline Correction with Asymmetric Least Squares Smoothing.
            This Python adaptation was taken from https://stackoverflow.com/a/50160920
        """
        self.tf_history.append(['AsLS', {"lam": lam, "p": p, "niter": niter}])
        if niter is None:
            niter = 20
        elif not isinstance(niter, int):
            niter = int(niter)

        for ii in range(self.spc.shape[0]):
            spectrum = self.spc.iloc[ii, :]
            self.spc.iloc[ii, :] = spectrum - self._get_AsLS_baseline(spectrum, lam, p, niter)


    def polyfit(self, order, niter=20, *args):
        """
        Fits a modified polyfit baseline for each spectrum and subtracts the baseline.

        :param order:
            Order of polynomial baseline
        :param niter:
            Number of iterations, default is 20. Increase if there are flat sections in the resulting spectrum
        :param args:
            None
        :return:
            None

        Reference:
            Lieber, C. A., and A. Mahadevan-Jansen, “Automated Method for Subtraction of Fluorescence from Biological
            Raman Spectra,” Applied Spectroscopy, 57, pp. 1363–1367 (2003).
            doi.org/10.1366/000370203322554518
        """
        import numpy as np

        self.tf_history.append(['polyfit', {"order": order, "niter": niter}])

        if niter is None:
            niter = 20

        def arrays_equal(a, b):
            if a.shape != b.shape:
                return False
            for ai, bi in zip(a.flat, b.flat):
                if ai != bi:
                    return False
            return True

        for ii in range(self.spc.shape[0]):
            spectrum = self.spc.iloc[ii, :]
            spc_fit = np.polyfit(spectrum.index, spectrum, order)
            baseline = np.zeros(spectrum.shape[0])
            baseline_current = np.polyval(spc_fit, spectrum.index)
            baseline_previous = np.zeros(baseline.shape)
            baseline_current = np.where(baseline_current < spectrum, baseline_current, spectrum)
            index = 0

            baseline_previous = np.zeros(baseline.shape)
            baseline_current = np.where(baseline_current < spectrum, baseline_current, spectrum)

            while not arrays_equal(baseline_previous, baseline_current) and index < niter:
                index += 1
                baseline_previous = baseline_current
                spc_fit = np.polyfit(spectrum.index, baseline_previous, order)
                baseline_current = np.polyval(spc_fit, spectrum.index)
                baseline_current = np.where(baseline_current < baseline_previous, baseline_current, baseline_previous)

            self.spc.iloc[ii, :] = spectrum - baseline_current

    def subtract(self, spectra, *args):
        """
        Subtracts each spectrum by a specified spectrum

        :param spectra: int
            Spectrum to use for the subtraction, numbering starting at 1
        :param args:
            None
        :return:
            None
        """
        self.tf_history.append(['subtract', {"spectra": spectra}])

        if spectra is None:
            print("Need to provide a integer for the spectra to subtract by.")
            return
        if spectra < 1 or spectra > self.spc.shape[0]+1:
            print(f"Spectral choice should be between 0 and {self.spc.shape[0]}")
            return
        if not isinstance(spectra, int):
            spectra = int(spectra)

        self.spc = self.spc.sub(self.spc.iloc[spectra-1, :], axis=1)

    def to_absorbance(self):
        """
        Converts the spectrum to an absorbance spectrum, assuming a transmittance or reflectance spectrum.

        :return:
            None

        Reference:
            Geladi, P., D. MacDougall, and H. Martens, “Linearization and Scatter-Correction for Near-Infrared
            Reflectance Spectra of Meat,” Applied Spectroscopy, 39, pp. 491–500 (1985).
        """

        self.spc = self.spc.apply(self._trans_to_absorbance, axis=1)

    def _trans_to_abs(self, y):
        """
        Internal function to transform from transmittance/reflectance to absorbance

        :param y: array-like
            Transmittance or reflectance spectrum

        :return: array-like
            Absorption spectrum

        Reference:
            Geladi, P., D. MacDougall, and H. Martens, “Linearization and Scatter-Correction for Near-Infrared
            Reflectance Spectra of Meat,” Applied Spectroscopy, 39, pp. 491–500 (1985).
        """
        import numpy as np

        return -np.log10(y)

    def _abs_to_trans(self, y):
        """
        Internal function to transform from absorbance to transmittance/reflectance

        :param y: array-like
            Absorption spectrum

        :return: array-like
            Reflectance/transmittance spectrum

        Reference:
            Geladi, P., D. MacDougall, and H. Martens, “Linearization and Scatter-Correction for Near-Infrared
            Reflectance Spectra of Meat,” Applied Spectroscopy, 39, pp. 491–500 (1985).
        """

        import numpy as np

        return np.power(10, 1-y)

    def _ref_to_KM(self, y):
        """
        Internal function to transform from reflectance to Kebulka-Munk

        :param y: array-like
            Reflectance spectrum

        :return:
            Kebulka-Munk transformed spectrum

        Reference:
            Geladi, P., D. MacDougall, and H. Martens, “Linearization and Scatter-Correction for Near-Infrared
            Reflectance Spectra of Meat,” Applied Spectroscopy, 39, pp. 491–500 (1985).
        """

        return (1 - y) ** 2 / 2 / y

    def _ref_to_invKM(self, y):
        """
        Internal function to transform from reflectance to inverse Kebulka-Munk

        :param y:array-like
            Reflectance spectrum

        :return:array-like
            inverse Kebulka-Munk spectrum

        Reference:
            Geladi, P., D. MacDougall, and H. Martens, “Linearization and Scatter-Correction for Near-Infrared
            Reflectance Spectra of Meat,” Applied Spectroscopy, 39, pp. 491–500 (1985).

        """
        import numpy as np

        return np.power(self._ref_to_KM(y), -1)

    def pareto(self, *args):
        """
            Performs in-place pareto scaling of data.

            Returns
            -------
            None.

            Reference:
                van den Berg RA, Hoefsloot HC, Westerhuis JA, Smilde AK, and van der Werf MJ. "Centering, scaling, and
                transformations: improving the biological information content of metabolomics data," BMC Genomics,
                2006 Jun 8;7:142. doi: 10.1186/1471-2164-7-142.
        """
        import numpy as np
        self.tf_history.append(['pareto', {}])
        self.spc = self.spc.sub(self.spc.mean(axis=1), axis=0).divide(np.sqrt(self.spc.std(axis=1)), axis=0)

    def pearson(self, u=4, v=3, *args):
        """
        Performs in-place Pearson's baseline correction of the data, with the baseline being approximated using a 4th
        order Legendre polynomial.

        Returns
        -------
        None.

        Reference:
            Pearson, G.A. "A general baseline-recognition and baseline-flattening algorithm." Journal of Magnetic
            Resonance, Aug 1997; Volume 27, Issue 2, pages 265-272. doi: 10.1016/0022-2364(77)90076-2
        """

        import numpy as np
        import math

        if u is None:
            u = 4
        if v is None:
            v = 3

        self.tf_history.append(['pearson', {"u": u, "v": v}])

        def heaviside(z):
            result = []
            for i in z:
                if i >= 0:
                    result.append(1)
                else:
                    result.append(0)

            return np.asarray(result)

        def is_correction_negligible(stddev, x):
            if all(np.abs(x) < (0.125 * stddev)):
                return True
            else:
                return False

        for ii in range(self.spc.shape[0]):
            print(f"Working on spectra {ii+1} of {self.spc.shape[0]}")

            Y = self.spc.iloc[ii, :].values

            sigma_0 = np.inf
            sigma_1 = 0
            g = np.ones(len(self.wav)) * np.inf

            while not is_correction_negligible(sigma_0, g):
                print("Correction not negligible")
                while sigma_0 != sigma_1:
                    print("sigma_1 != sigma0")
                    print(f"sigma_0 = {sigma_0}")
                    sigma_1 = np.std(Y * heaviside(u*sigma_0 - np.abs(Y)))
                    print(f"sigma_1 = {sigma_1}")
                    sigma_0 = sigma_1
                print("sigma_1 == sigma_0")

                print("Fitting 4th order Legendre polynomial")
                fitted = np.polynomial.legendre.legfit(self.wav.values, Y, deg=3)

                print("Calculating g(x)")
                g = np.polynomial.legendre.legval(self.wav.values, fitted)

                print("Subtracting g(x) from the spectrum")
                Y = Y - g
                self._baselines.iloc[ii, :] += g

            self._baselines.iloc[ii, :] = g
            self.spc.iloc[ii, :] = Y

    def export_csv(self, f_path):
        self.spc.transpose().to_csv(f_path)

    def export_excel(self, f_path):
        import pandas as pd

        with pd.ExcelWriter(f_path) as writer:
            self.spc.transpose().to_excel(writer, sheet_name="Processed Spectra")
            self._spc_raw.transpose().to_excel(writer, sheet_name="Raw Spectra")
            self._baselines.transpose().to_excel(writer, sheet_name="Baselines")
            self.war.to_excel(writer, sheet_name='Perturbation')
            tfs = pd.DataFrame(self.tf_history)
            tfs.to_excel(writer, sheet_name="Processing Recipe")