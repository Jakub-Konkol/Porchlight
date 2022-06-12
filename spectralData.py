# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:23:25 2022

@author: Jakub
"""


class SpectralData():
    def __init__(self, file=None):
        import numpy as np
        import pandas as pd
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
            store_spc = np.array([])
            self.file_source = []
            self.tf_history = []

            # start with first file
            if isinstance(file, str):
                wav, spc = self.getDataFromFile(file)
            else:
                wav, spc = self.getDataFromFile(file[0])

            if spc.ndim == 1:
                store_spc = np.concatenate((store_spc, spc))
            elif spc.ndim == 2:
                store_spc = np.concatenate((store_spc, spc[:, 0]))
                store_spc = np.column_stack((store_spc, spc[:, 1:]))

            self._wav_raw = wav
            self.wav = self._wav_raw

            if not isinstance(file, str) and len(file) > 1:
                for ii in range(1, len(file)):
                    wav, spc = self.getDataFromFile(file[ii])
                    store_spc = np.column_stack((store_spc, spc))
                    # TODO: what if wav is not the same? some kind of pandas merge

            self._spc_raw = pd.DataFrame(store_spc, index=self.wav)

            self.spc = self._spc_raw
            self.war = np.zeros((self.spc.shape[1]))

    def getDataFromFile(self, file):
        """
        Perform logic to determine filetype and corresponding import function. Import data and append file to imported_file array to keep track.

        Parameters
        ----------
        file : STR
            Path to file.

        Returns
        -------
        wav : NP.ARRAY
            Array of independent wavenumber values
        spc : NP.ARRAY
            Array of spectral values, ordered as one spectra per column

        """
        import os
        ext = os.path.splitext(file)[-1].lower()
        # go through decision tree to get correct import function
        if ext == '.txt':
            wav, spc = self.readTextFile(file)
        elif ext == ".spc":
            wav, spc = self.readSPCFile(file)
        elif ext == '.jdx':
            print("J-Camp file import not implemented yet")
            # TODO: figure out j-camp import
        elif ext == '.csv':
            wav, spc = self.readCSVFile(file)
        else:
            print("This filetype is not supported")

        if spc.ndim == 1:
            self.file_source.append(os.path.split(file)[1])
        elif spc.ndim == 2:
            self.file_source = self.file_source + [os.path.split(file)[1] for x in range(spc.shape[1])]

        return wav, spc

    def readTextFile(self, file):
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
        data = np.loadtxt(file)
        # assuming that there are more data points than spectra, make columnwise
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        wav = data[:, 0]
        spc = data[:, 1:]

        # return squeezed spc in case only 1D
        return wav, np.squeeze(spc);

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
        return wav, np.squeeze(spc);

    def readSPCFile(self, file):
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
        spc = np.array(temp).transpose()
        return wav, np.squeeze(spc)

    def plot(self):
        self.spc.plot()

    def reset(self):
        """
        Reset the spectra and wavenumber to raw imported values.
        """
        self.spc = self._spc_raw
        self.wav = self._wav_raw

    def rolling(self, window):
        if window is None:
            print("Warning: no window length supplied. Defaulting to 1.")
            window = 1
        if not isinstance(window, int):
            window = int(window)

        # first perform the rolling window smooth
        self.spc = self.spc.rolling(window).mean()

        # use the NANs to cut the wav and spc matrices
        self.wav = self.wav[self.spc.iloc[:, 0].notna()]
        self.spc = self.spc.dropna()

    def SGSmooth(self, window, poly):
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

        self.spc = pd.DataFrame(savgol_filter(self.spc, window, poly, axis=0), columns=self.spc.columns,
                                index=self.spc.index)

    def SGDeriv(self, window, poly, order):
        from scipy.signal import savgol_filter
        import pandas as pd

        if not isinstance(window, int):
            window = int(window)
        if not isinstance(poly, int):
            poly = int(poly)

        self.spc = pd.DataFrame(savgol_filter(self.spc, window, poly, order, axis=0), columns=self.spc.columns,
                                index=self.spc.index)

    def snv(self):
        """
        Performs in-place SNV normalization of data.

        Returns
        -------
        None.

        """
        import numpy as np
        self.spc = self.spc.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)

    def msc(self, reference=None):
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

        if reference == None:
            ref = self.spc.mean(axis=1)
        else:
            if isinstance(reference, int):
                ref = self.spc.iloc[:, reference]
            else:
                reference = int(reference)
                ref = self.spc.iloc[:, reference]

        for ii in range(self.spc.shape[1]):
            fit = np.polyfit(ref, self.spc.iloc[:, ii], 1, full=True)
            self.spc.iloc[:, ii] = (self.spc[ii] - fit[0][1]) / fit[0][0]

    def trim(self, start, end):
        if start is None:
            start = self.spc.index[0]
        if end is None:
            end = self.spc.index[-1]

        self.spc = self.spc.loc[(self.spc.index > start) & (self.spc.index < end)]
        self.wav = self.wav[(self.wav > start) & (self.wav < end)]

    def invtrim(self, start, end):
        if start is None:
            start = self.spc.index[0]
        if end is None:
            end = self.spc.index[-1]

        self.spc = self.spc.loc[(self.spc.index < start) | (self.spc.index > end)]
        self.wav = self.wav[(self.wav < start) | (self.wav > end)]

    def area(self):
        import numpy as np
        self.spc = self.spc / self.spc.apply(lambda x: np.trapz(x, self.wav), axis=0)

    def lastpoint(self):
        self.spc = self.spc.sub(self.spc.iloc[-1, :])

    def mean_center(self):
        self.spc = self.spc.sub(self.spc.mean(axis=0))

    def peaknorm(self, wavenumber):
        index = self.spc.index.get_loc(wavenumber, method='nearest')
        self.spc = self.spc.divide(self.spc.iloc[index, :])

    def vector(self):
        self.spc = self.spc.divide(((self.spc ** 2).sum(axis=0)) ** (1 / 2))

    def minmax(self, min_val=0, max_val=1):
        self.spc = min_val + (self.spc.sub(self.spc.min(axis=0))) * (max_val - min_val) / (
                    self.spc.max(axis=0) - self.spc.min(axis=0))

    def _get_AsLS_baseline(self, y, lam, p, niter):
        # adapted from https://stackoverflow.com/a/50160920
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        L = self.spc.shape[0]
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z


import os

# import cProfile

# files = os.listdir("spectra")
# files = ["./spectra/" + x for x in files]
# testData = SpectralData(files)

#file_spc = 'C:/Users/Jakub/OneDrive/Documents/Rutgers/Tsilomelekis/Data/Bioreactor/Bioreactor/P1BR2/Probe 1_P1BR2 1075 15m_20191112-080528/P1BR2 1075 15m_20191112-081817.spc'
#spc_test = SpectralData((file_spc,))

# nirData = SpectralData(("C:/Users/Jakub/Desktop/NIRSpectra_noHeader.txt",))
