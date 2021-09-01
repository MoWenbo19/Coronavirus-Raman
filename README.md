# Coronavirus-Raman
This database include the code and the data that support the findings of Rapid classification of coronaviruses by Raman spectroscopy based on deep learning.  
In /Code/, Python scripts are available to train and classify coronavirus Raman spectral datasets using MLP, CNN, and ResNet models.  
In /Data/, Data is packaged according to 3-class identification task and 5-class task.   
In /Data/3-class, there are Raman spectrum data files for MERS-CoV, SARS-CoV, and SARS-CoV-2. Each has 14 files. The five files whose names do not contain “map” are single-point collection. The file whose name ends with map0 is a 20 × 20 discrete spot map. The rest are eight 15 × 15 discrete spot maps, for a total of 2205 spectra.  
In /Data/5-class, there are Raman spectrum data files for MERS-CoV, SARS-CoV, SARS-CoV-2, HCoV-HKU1, and HCoV-OC43. Each has 6 files. The five files whose names do not contain “map” are single-point collection. The file whose name ends with map0 is a 20 × 20 discrete spot map, for a total of 405 spectra.  
