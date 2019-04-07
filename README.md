# pan18ap (PAN 2018, Author Profiling task)
The participation of the *Natural Language Processing Lab* of the *University of Ottawa*\
in the [Author Profiling shared task at PAN 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html)

Our model was the **best-performing** model in *textual classification*, with the accuracy of 0.8221, 0.82, and 0.809 on the English, Spanish, and Arabic datasets respectively. Considering the combination of textual and image classification, and all three datasets, *our model ranked __second__ in the task* out of 23 teams.

Our approach to gender identification in Twitter, takes advantage of textual information solely, and consists of tweet preprocessing, feature construction, dimensionality reduction using Latent Semantic Analysis (LSA), and classification model construction. We propose a linear Support Vector Machine (SVM) classifier, with different types of word and character n-grams as features.


## *Contents*
1. [Citation](#citation)
1. [Motivation](#motivation)
1. Getting started: a beginner's guide to PAN shared tasks
    1. [I am looking to participate in a shared task at PAN. Where do I start?](#i-am-looking-to-participate-in-a-shared-task-at-pan-where-do-i-start)
    1. [I am new to Python. What do I do?](#i-am-new-to-python-what-do-i-do)
1. Installation
    1. [Requirements](#requirements)
    1. [Dataset](#dataset)
1. [Support](#support)
1. [Acknowledgments](#acknowledgments)


## Citation
If our code comes in useful to you, please don't forget to cite our paper:
> Daneshvar, S., & Inkpen, D. (2018). [*Gender Identification in Twitter using N-grams and LSA*](https://scholar.google.com/scholar?cluster=4499254726211723674). Notebook for PAN at CLEF 2018. CEUR Workshop Proceedings, 2125, 1–10. &nbsp; [**`Publisher`**](http://ceur-ws.org/Vol-2125/) &nbsp;[**`Paper PDF`**](http://ceur-ws.org/Vol-2125/paper_213.pdf) &nbsp;[**`Cite: BibTeX`**](../../raw/master/Daneshvar2018.bib)


## Motivation
You are probably here for one of the following reasons:
- You are a participant in a [shared task at PAN](https://pan.webis.de/tasks.html), looking for approaches that have worked well for other participants of the task in the previous years.
- You are a *machine learning* and *natural language processing* enthusiast, looking for some starting code to try out some NLP and ML experiments.

In the next section, I will give step by step instructions on how to reproduce the results of the previous year's participants. These instructions are geared towards those with little or no experience with NLP and ML research. If this does not apply to you or you are not looking to participate at a PAN shared task, you may skip the next section. However, I still encourage you to take a look at [our paper](#citation) for full details on our approach.


## Getting started: a beginner's guide to PAN shared tasks
### I am looking to participate in a shared task at PAN. Where do I start?
In a nutshell, begin with reproducing the experiments of the previous year's participants, and then try to improve the results by implementing new methods.

Here, I will specifically explain the Author Profiling task at PAN 2018, but the following steps apply to pretty much any shared task.
1. First, head to the [Shared Tasks @ PAN](https://pan.webis.de/tasks.html) webpage, browse the tasks and register for the tasks that you like. It's free!
1. You will receive an email from the organizers explaining how to access the dataset, how to submit your code (on the [TIRA](https://www.tira.io) virtual machine), and so on.
1. Head back to the shared task's webpage, and browse the prevoius year's task: [Author Profiling task, PAN @ CLEF 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html)\
Here, you can see the ranking of the participating teams. Our team (named `daneshvar18`) ranked **second** in the *global ranking* and gained the **highest accuracy results** among all 23 participating teams in *textual classification*.\
On the same page, you can also find the following paper:
    > Francisco Rangel, Paolo Rosso, Martin Potthast, Benno Stein. [*Overview of the 6th Author Profiling Task at PAN 2018: Multimodal Gender Identification in Twitter*](http://ceur-ws.org/Vol-2125/invited_paper_15.pdf). In: CLEF 2018 Labs and Workshops, Notebook Papers. CEUR Workshop Proceedings. CEUR-WS.org, vol. 2125.
    
    Start by reviewing this paper, to get a sense of what approaches have been tried in the previous year and what has worked best for other participants. The next step would be to reproduce the experiments of other participants.
1. Head to the [PAN's publications](https://pan.webis.de/publications.html) page (not yet updated with the 2018 papers) or the [CLEF 2018 Working Notes](http://ceur-ws.org/Vol-2125/), and look for the notebook papers of the top-ranking teams.
The papers should include all the required details to reproduce their experiments. The links to our paper are provided [here](#citation).
1. Some teams also publish their source code. You can find them at [PAN's GitHub](https://github.com/pan-webis-de). To explore our code, continue to the [Installation](#installation) section.

### I am new to Python. What do I do?
A great place to start is the [beginner's guide](https://www.python.org/about/gettingstarted/) and the [Python tutorial](https://docs.python.org/3/tutorial/) at Python's own website. Needless to say, there are plenty of books, online courses and GitHub repositories to help you learn Python, but we won't get into that here.

Once you have your Python installation up and running, you can continue to the [next section](#installation), to set up my code on your computer.

In my code, I have tried to include some additional explanations and references in the comments, that would make things a bit more clear to a novice programmer.


## Installation
### Requirements
The following packages are used in this project:
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [NLTK](https://www.nltk.org/install.html)
- [Matplotlib](https://matplotlib.org/users/installing.html)

You can install all three packages using *Conda*:
```
conda install scikit-learn
conda install nltk
conda install matplotlib
```

Or using *pip*:
```
pip install -U scikit-learn
pip install -U nltk
pip install -U matplotlib
```

For specific installation instructions refer to the links above.

### Dataset
You can download the training corpus from the shared task's webpage: [Author Profiling, PAN @ CLEF 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html)

You can even download the test corpus—this is the corpus that our programs were tested against at the time of submission on the TIRA virtual machine. At the time, we did not have access to these files.

Note that the ZIP files are password-protected. To obtain the passwords, [contact the organizers of PAN](https://pan.webis.de/contact.html). They are really helpful and will give you useful advice.

Once you have downloaded and extracted the *PAN 2018 Author Profiling* training corpus, add it to the `data/` directory as follows:
```
data/
│   Flame_Dictionary.txt
│
└───PAN 2018, Author Profiling/
    ├───ar/
    ├───en/
    └───es/
```

Each of the `ar`, `en` and `es` folders in the provided corpus are structured as follows:
```
en/
│   en.txt
│
├───photo/
│   └───(3000 folders, each containing 10 photos)——You may skip these files and folders, as we won't use them.
└───text/
        (3000 XML files)
```

Please note that you may skip copying the `photo/` folders to the project's directory, since our code only uses the textual data and does not use any of the photos in the dataset.

Moreover, I have included the [Flame dictionary](http://www.site.uottawa.ca/~diana/resources/) in the `data/` directory in the repository, but you will not need it to reproduce our results, since we did not end up using the *Flame dictionary* in our final model.

The program should now be ready to run. For more information, refer to the docstrings.


## Support
I hope you will find this information useful. If you have any specific questions about the code or our approach that you cannot find the answer to in the code comments or in the notebook paper, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/samandaneshvar/) or send me an email. You can find my email address on [our paper](#citation).


## Acknowledgments
- Thanks to the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and the **University of Ottawa** for funding this research.
- Thanks to the organizing committee of PAN, especially Paolo Rosso, Francisco Rangel and Martin Potthast for their encouragement and kind support.
