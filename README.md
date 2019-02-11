# PAN18_AuthorProfiling
The participation of the Natural Language Processing Lab of the University of Ottawa in the author profiling shared task at PAN 2018

Our approach to gender identification in Twitter performed on the tweet corpus provided by CLEF for the task, takes advantage of textual information solely, and consists of tweet preprocessing, feature construction, dimensionality reduction using Latent Semantic Analysis (LSA), and classification model construction. We propose a linear Support Vector Machine (SVM) classifier, with different types of word and character n-grams as features.

Our model was the best-performing model in textual classification, with the accuracy of 0.8221, 0.82, and 0.809 on the English, Spanish, and Arabic datasets respectively. Considering the combination of textual and image classification, and all three datasets, our model ranked second in the task.

If this code is helpful, please don't forget to cite our paper:
https://scholar.google.com/scholar?cluster=4499254726211723674

## Setup
The following three folders are set to be ignored by Git:
* data/
* logs/
* pickles/

Add the PAN 2018 Author Profiling training corpus and the Flame Dictionary to the "data/" directory as follows:
```
data/  
  ar/  
  en/  
  es/  
  Flame_Dictionary.txt
```

Each of the "ar", "en" and "es" folders are structured as follows:
```
en/
  photo/
    (3000 folders, each containing 10 photos)
  text/
    (3000 XML files)
  en.txt
```
