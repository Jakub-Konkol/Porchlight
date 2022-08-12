python --version || Python is not installed
ECHO Installing packages as specified in requirements.txt
python -m pip install -r requirements.txt || Error running pip installation
ECHO Installation complete!