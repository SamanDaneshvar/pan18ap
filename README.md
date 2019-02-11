# pan18ap (PAN 2018, Author Profiling task)
The participation of the Natural Language Processing Lab of the University of Ottawa in the [author profiling shared task at PAN 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html)

Our model was the **best-performing** model in *textual classification*, with the accuracy of 0.8221, 0.82, and 0.809 on the English, Spanish, and Arabic datasets respectively. Considering the combination of textual and image classification, and all three datasets, *our model ranked __second__ in the task* out of 23 teams.

Our approach to gender identification in Twitter performed on the tweet corpus provided by CLEF for the task, takes advantage of textual information solely, and consists of tweet preprocessing, feature construction, dimensionality reduction using Latent Semantic Analysis (LSA), and classification model construction. We propose a linear Support Vector Machine (SVM) classifier, with different types of word and character n-grams as features.

## Citation
If this code is helpful to you, please don't forget to cite our paper:
> Daneshvar, S., & Inkpen, D. (2018). [*Gender Identification in Twitter using N-grams and LSA*](https://scholar.google.com/scholar?cluster=4499254726211723674). Notebook for PAN at CLEF 2018. CEUR Workshop Proceedings, 2125, 1–10. [[Publisher](http://ceur-ws.org/Vol-2125/)] [[Paper PDF](http://ceur-ws.org/Vol-2125/paper_213.pdf)] [[Cite: BibTeX](../../raw/master/Daneshvar2018.bib)]


## Motivation
You are probably here for one of the following reasons:
- You are a participant in a [shared task at PAN](https://pan.webis.de/tasks.html), looking for approaches that have worked well for other participants of the task in the previous years.
- You are a *machine learning* and *natural language processing* enthusiast, looking for some starting code to try out some NLP and ML experiments.

In the next section, I will give you step by step instructions on how to reproduce the results of the other teams who participated in a shared task at PAN in the previous years. If you are not looking to participate at a PAN shared task, you may skip the next section, but I still encourage you to take a look at [our paper](#citation) for full details on our approach.


## Getting started
### I am looking to participate at a shared task at PAN. Where do I start?
1. First, head to the [Shared Tasks @ PAN](https://pan.webis.de/tasks.html) webpage, browse the tasks and register for the tasks that you like. It's free!
1. You will receive an email from the organizers explaining how to access the dataset, how to submit your code (on the [TIRA](https://www.tira.io) virtual machine), and so on.
1. Head back to the shared task's webpage, and browse the prevoius year's task: [Author Profiling task, PAN @ CLEF 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html).

## Installation
### Requirements

### Dataset

Add the PAN 2018 Author Profiling training corpus and the Flame Dictionary to the `data/` directory as follows:
```
data/
│   Flame_Dictionary.txt
│
└───PAN 2018, Author Profiling/
    ├───ar/
    ├───en/
    └───es/
```

Each of the `ar`, `en` and `es` folders are structured as follows:
```
en/
│   en.txt
│
├───photo/
│   └───(3000 folders, each containing 10 photos)
└───text/
        (3000 XML files)
```

## Support

## Acknowledgments
