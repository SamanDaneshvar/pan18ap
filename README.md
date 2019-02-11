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


## Getting started: a beginner's guide to PAN shared tasks
### I am looking to participate in a shared task at PAN. Where do I start?
Here, I will specifically explain the Author Profiling task at PAN 2018, but the following steps apply to pretty much any shared task.
1. First, head to the [Shared Tasks @ PAN](https://pan.webis.de/tasks.html) webpage, browse the tasks and register for the tasks that you like. It's free!
1. You will receive an email from the organizers explaining how to access the dataset, how to submit your code (on the [TIRA](https://www.tira.io) virtual machine), and so on.
1. Head back to the shared task's webpage, and browse the prevoius year's task: [Author Profiling task, PAN @ CLEF 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html).
Here, you can see the ranking of the participating teams. Our team was named `daneshvar18`, and you can see that we ranked second in the global ranking, with the highest accuracy results in textual classification.
You can also find the following paper:
    > Francisco Rangel, Paolo Rosso, Martin Potthast, Benno Stein. [*Overview of the 6th author profiling task at pan 2018: multimodal gender identification in Twitter*](http://ceur-ws.org/Vol-2125/invited_paper_15.pdf). In: CLEF 2018 Labs and Workshops, Notebook Papers. CEUR Workshop Proceedings. CEUR-WS.org, vol. 2125.
    
    Start by reviewing this paper, to get a sense of what approaches have been tried in the previous year and what has worked best for other participants.
1. Head to the publications page for that year of the task at [PAN](https://pan.webis.de/publications.html) (not updated with the 2018 papers, for some reason) or [CEUR](http://ceur-ws.org/Vol-2125/), and look for the notebook papers of the top-ranking teams.
The papers should include all the required details to reproduce their experiment results. The links to our paper are provided [here](#citation).
1. Some teams also publish their source code. You can find them at [PAN's GitHub](https://github.com/pan-webis-de).

### I am new to Python. What do I do?
A great place to start is the [beginner's guide](https://www.python.org/about/gettingstarted/) and the [Python tutorial](https://docs.python.org/3/tutorial/) at Python's own website.

Once you have your Python installation up and running, you can continue to the next section, to set up my code on your computer.

In my code, I have tried to include some additional comments, explanations, and references that would make things a bit more clear to a beginner.

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
