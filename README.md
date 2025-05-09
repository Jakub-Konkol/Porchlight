# Porchlight

Porchlight is a preprocessing aid for spectroscopists and for students to learn preprocessing.

![Porchlight Logo](/repo-resources/porchlight.png)

This program was published as part of the article [Porchlight: An Accessible and Interactive Aid in Preprocessing of Spectral Data](https://pubs.acs.org/doi/10.1021/acs.jchemed.2c00812).

## Features

Porchlight allows you to open a variety of spectral files and immediately visualize the preprocessing you perform on your spectra.
This way, you can get a better intuition for the impact various preprocessing methods have.
The software is also designed for the classroom as well, with important features to support learning such as dynamic plot axis labelling.

![GUI in use](/repo-resources/gui.PNG)

## Installation
<!---
Porchlight can be installed using

`pip install porchlight`

If you would like to include optional packages for specific filetypes, use the following:

`pip install porchlight[SPC

someone sniped the package name porchlight from me on the pipy index

--->
To install the most up-to-date version from GitHub, you can use the following:

`pip install git+https://github.com/Jakub-Konkol/Porchlight.git`

## Use
This section describes the use of Porchlight, either through the GUI or as a class you can use in your scripts.

### GUI
To invoke the GUI, call the following from the terminal

`python -m porchlight`

### Within scripts

The data handling backend code be called in scripts using

```python
from porchlight.spectralData import SpectralData
```


From here, one can instantiate the class by providing a list of file directories containing spectral data, either in CSV, TXT, or Thermo-Gram SPC (if optional `spc-spectra` is installed).

```python
data = SpectralData(['myFile.csv'])
```

All preprocessing techniques are methods of this class, which are performed in-place. To perform SNV, one can use

```python
data.reset()
data.SNV()
```

We recommend performing a reset at the start, to make sure you don't accidentally process processed data.

### Class Variables

`wav` - numpy array containing the abscissa axis

`war` - pandas DataFrame containing labels for each spectrum (not currently used)

`tf_history` - a list that describes the preprocessing steps performed to the spectra. Each list element is a list where
the first element is the preprocessing step and the second element is a dictionary with the values given for the method.

`file_source` - a list of directories that describe the source file for each spectrum. If one file provides multiple
spectra, then it is repeated multiple times.

`spc` - a pandas DataFrame containing the spectral data in a tidy format. Each row is one spectrum, each column 
is a corresponding abscissa value. The column names are equivalent to the variable `wav`.

`_baselines` - a pandas DataFrame with calculated baselines, if one was calculated.

# Acknowledgements

This material is based upon work supported in part by Rutgers, The State University of New Jersey, and the National 
Science Foundation Award 1751683.

This software has been developed by [Jakub Konkol](https://jakubkonkol.com/) of the 
[Dr. Tsilomelekis](https://www.gtsilomelekis.com/) group at Rutgers, the State University of New Jersey. 
[Come see our research!](https://www.gtsilomelekis.com/)

:copyright: 2021-2025 Rutgers, the State University of New Jersey
