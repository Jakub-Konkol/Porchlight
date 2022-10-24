Porchlight
==========

Porchlight is a preprocessing aid for spectroscopists and for students to learn preprocessing.

Features
--------

Porchlight allows you to open a variety of spectral files and immediately visualize the preprocessing you perform on your spectra.
This way, you can get a better intuition for the impact various preprocessing methods have.
The software is also designed for the classroom as well, with important features to support learning such as dynamic plot axis labelling.


Installation
------------
Porchlight can be installed using

`pip install porchlight`

If you would like to include optional packages for specific filetypes, use the following:

`pip install porchlight[spc-spectra]`

To install the most up-to-date version from GitHub, you can use the following:

`pip install git+https://github.com/Jakub-Konkol/Porchlight.git`

Use - GUI
---------
To invoke the GUI, call the following from the terminal

`python -m porchlight`

Use - Within scripts
-------------

The data handling backend code be called in scripts using

`from porchlight.spectralData import SpectralData`.

From here, one can instantiate the class by providing a list of file directories containing spectral data, either in CSV, TXT, or Thermo-Gram SPC (if optional `spc-spectra` is installed).

`data = SpectralData(['myFile.csv']`

All preprocessing techniques are methods of this class, which are performed in-place. To perform SNV, one can use

`data.reset()`

Data is stored in a tidy format as spc, which is a pandas DataFrame with column headers being the abscissa and the rows containing the ordinate intensities, each spectrum.

Acknowledgements
----------------

This material is based upon work supported in part by Rutgers, The State University of New Jersey, and the National Science Foundation Award 1751683.

This software has been developed by [Jakub Konkol](https://jakubkonkol.com/) of the [Dr. Tsilomelekis](https://www.gtsilomelekis.com/) group at Rutgers, the State University of New Jersey. [Come see our research!](https://www.gtsilomelekis.com/)

:copyright: 2021-2022 Rutgers, the State University of New Jersey