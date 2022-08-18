# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:23:25 2022

@author: Jakub
"""


class SpectralData():
    def __init__(self, file=None):
        import numpy as np
        import pandas as pd
        import copy
        import re
        import os

        if file is None:
            self._wav_raw = [0]
            self._spc_raw = [0]
            self.war = [0]
            self.spc = [0]
            self.wav = [0]
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
                    # TODO: what if wav is not the same? some kind of pandas merge

            combined = pd.concat(store_spc, axis=0, ignore_index=True)
            combined.sort_index(axis=1, inplace=True)

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
            self.war = np.zeros((self.spc.shape[1]))
            # self.baseline = pd.DataFrame(np.zeros(self._spc_raw.shape), columns=self._wav_raw)

    def getDataFromFile(self, file):
        """
        Perform logic to determine filetype and corresponding import function. Import data and append file to imported_file array to keep track.

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
            # raw.iloc[:,0] = round(raw.iloc[:,0],2)
            raw = raw.set_index(raw.columns[0]).transpose()
        else:
            raw = raw.transpose().set_index(raw.columns[0]).transpose()

        # return squeezed spc in case only 1D
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

    def plot(self):
        self.spc.transpose().plot()

    def reset(self, *args):
        """
        Reset the spectra and wavenumber to raw imported values.
        """
        # import pandas as pd
        # import numpy as np
        self.spc = self._spc_raw.copy(deep=True)
        self.wav = self._wav_raw.copy(deep=True)
        #self.baseline = pd.DataFrame(np.zeros(self._spc_raw.shape), columns=self._wav_raw)

    def rolling(self, window, *args):
        if window is None:
            print("Warning: no window length supplied. Defaulting to 1.")
            window = 1
        if not isinstance(window, int):
            window = int(window)

        self.spc = self.spc.transpose()
        # first perform the rolling window smooth
        self.spc = self.spc.rolling(window).mean()

        # use the NANs to cut the wav and spc matrices
        self.wav = self.wav[self.spc.iloc[:, 0].notna()]
        self.spc = self.spc.dropna()

        self.spc = self.spc.transpose()

    def SGSmooth(self, window, poly, *args):
        """
        Performs Savitzky-GOlay smoothing of the data as implemented by scipy.signal.

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

        self.spc = pd.DataFrame(savgol_filter(self.spc, window, poly, axis=1), columns=self.spc.columns,
                                index=self.spc.index)

    def SGDeriv(self, window, poly, order, *args):
        from scipy.signal import savgol_filter
        import pandas as pd

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

        """
        import numpy as np
        self.spc = self.spc.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=1)

    def msc(self, reference=None, *args):
        """
        Performs in-place MSC scatter correction of data.

        Parameters
        ----------
        reference : INT, optional
            The value of the spectra to be fitted to. If none is provided, the average spectrum is used. The default is None.

        Returns
        -------
        None.

        """
        import numpy as np

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

    def trim(self, start=None, end=None, *args):
        if start is None:
            start = self.spc.columns[0]
        if end is None:
            end = self.spc.columns[-1]

        self.spc = self.spc.transpose()
        self.spc = self.spc.loc[(self.spc.index > start) & (self.spc.index < end)]
        self.wav = self.wav[(self.wav > start) & (self.wav < end)]
        self.spc = self.spc.transpose()

    def invtrim(self, start=None, end=None, *args):
        if start is None:
            start = self.spc.index[0]
        if end is None:
            end = self.spc.index[-1]

        self.spc = self.spc.transpose()
        self.spc = self.spc.loc[(self.spc.index < start) | (self.spc.index > end)]
        self.wav = self.wav[(self.wav < start) | (self.wav > end)]
        self.spc = self.spc.transpose()

    def area(self, *args):
        import numpy as np
        self.spc = self.spc.divide(self.spc.apply(lambda x: np.trapz(x, self.wav), axis=1), axis=0)

    def lastpoint(self, *args):
        self.spc = self.spc.sub(self.spc.iloc[:, -1], axis=0)

    def mean_center(self, option=False, *args):
        if not option:
            self.spc = self.spc.sub(self.spc.mean(axis=1), axis=0)
        elif option:
            self.spc = self.spc.sub(self.spc.mean(axis=0), axis=1)

    def peaknorm(self, wavenumber, *args):

        self.spc = self.spc.transpose()
        index = self.spc.index.get_loc(wavenumber, method='nearest')
        self.spc = self.spc.divide(self.spc.iloc[index, :])
        self.spc = self.spc.transpose()

    def vector(self, *args):
        self.spc = self.spc.divide(((self.spc ** 2).sum(axis=1)) ** (1 / 2), axis=0)

    def minmax(self, min_val=0, max_val=1, *args):

        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 1

        self.spc = self.spc.transpose()
        self.spc = min_val + (self.spc.sub(self.spc.min(axis=0))) * (max_val - min_val) / (
                self.spc.max(axis=0) - self.spc.min(axis=0))
        self.spc = self.spc.transpose()

    def _get_AsLS_baseline(self, y, lam, p, niter):
        # adapted from https://stackoverflow.com/a/50160920
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
        if niter is None:
            niter = 20
        elif not isinstance(niter, int):
            niter = int(niter)

        for ii in range(self.spc.shape[0]):
            spectrum = self.spc.iloc[ii, :]
            self.spc.iloc[ii, :] = spectrum - self._get_AsLS_baseline(spectrum, lam, p, niter)


    def polyfit(self, order, niter=20, *args):
        import numpy as np
        import pandas as pd

        if niter is None:
            niter = 20

        def arrays_equal(a, b):
            if a.shape != b.shape:
                return False
            for ai, bi in zip(a.flat, b.flat):
                if ai != bi:
                    return False
            return True

        # self.baseline = pd.DataFrame(np.zeros(self.spc.shape), columns=self.wav)

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

            # self.baseline.iloc[ii, :] = baseline_current
            self.spc.iloc[ii, :] = spectrum - baseline_current

    def subtract(self, spectra, *args):
        if spectra is None:
            print("Need to provide a integer for the spectra to subtract by.")
            return
        if spectra < 1 or spectra > self.spc.shape[0]+1:
            print(f"Spectral choice should be between 0 and {self.spc.shape[0]}")
            return
        if not isinstance(spectra, int):
            spectra = int(spectra)

        self.spc = self.spc.sub(self.spc.iloc[spectra-1, :], axis=1)
