# OxleyPython
These are the source files of the implementation of the Oxley's machining model using the LMFIT library in Python. This work is related to the PhD thesis of Maxime Dawoua Kaoutoing:

## Usage

All code is in the **ExtOxley_LMFIT.py** file. execution of the code is done using the following command:

	python3 ExtOxley_LMFIT.py

This computes the output values depending on the cutting conditions defined at the beginning of the Python file.

All graphs generated end included in this package can be generated by un-commenting all the piece of code located in the 3 sections numbered from 1 to 3 at the end of the main program.

* Section 1 : generates the graphs of the evolution of 1 internal parameter vs. another one
* Section 2 : Sensivity study
* Section 3 : Tests the unicity of the solution by running 15625 runs

***
Olivier Pantalé  
Full Professor of Mechanics  
email : olivier.pantale@enit.fr

Laboratoire Génie de Production  
Ecole Nationale d'Ingénieurs de Tarbes  
Université de Toulouse  
47 Avenue d'Azereix - BP 1629  
65016 TARBES - CEDEX - FRANCE