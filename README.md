# NPR--Number_Plate_Recognition

This is my course project for Pattern Recoginition. 
The goal is to develop a Number Plate Recognition System, which could tell what's the number plate on the input image. 
Its not of perfect result now, but its easy to get started if you are interested.

To demo this Project:
  - run npr_clf.py
  - enter the image names in directory 'demoData', 'demoData/NP_image1.jpg' for instance(the outcome does include noise).
  - if you want to visualize the process, set the 'showProcess = True'  for function demo_NPR.   
   
-------------------------------------------------------

The whole project could be divided into:
- Preprocessing             -- preProcess.py
  - plate localization
  - character segmentation
- Feature Extraction        -- featureExt.py
- Plate Reconition          -- npr_clf.py

Process Illustration

![Alt text](./process_illustration.png?raw=true "Title")
-------------------------------------------------------
_MORE DETAIL_

__PreProcessing__
- We use the double-phase statistical analysis to achieve plate localization, which is based on the frequence variance. 
- after located the plcate, we clip it out, do the character segmentation. 

__Featrue Extartion__
- The features we used for this Project is character edge:
  - 6 regions per character
  - 8 edge types,
  - feature vector length: 6*8
  
__Clissifier__
  - we use the KNN classifier. 
 
-------------------------------------------------------
The project basically follow this book:
http://javaanpr.sourceforge.net/anpr.pdf
