*PROGGERS*
==========

As the size of data sets constantly grows, users of exploratory data analysis
systems have to wait longer and longer for monolithic implementations to return
results. This impedes the usual approach of ad-hoc exploratory analysis, as
non-fluent user interactions will negatively impact the data analyst's ability
to focus. To tackle that problem, progressive visual
analytics (PVA) emerged recently as a new paradigm. In this work, we preset a
back-end only PVA system called *PROGGERS*. It is based on a progressive 
adaptation of the approximate query processing system *IDEA* proposed in 2017
by Galakatos et. al. \[1\]. Furthermore, we extend the system to support a wider
range of queries. Our experiments show that one of the core concepts, the
*tail index*, is capable of noticeably speeding up the system for queries that
filter by rare subpopulation which were involved in previous queries of the data
exploration session.

This repository contains the code of my master's thesis. Which can be found at
[https://example.com/yet_to_come](https://example.com/yet_to_come).
The code contains doc strings and comments. For a broader overview of the
context, refer to the thesis.


### Repository Structure
The code for *PROGGERS* can be found in the folder with the same name.  
The code for the experiments ran for the evaluation lies in the project's root.  
The raw data and plots of the experiment can be found in `out_results`.


### How to Make it Run
Install **Python 3.6** or newer. Then install the requirements:
`pip install -r proggers/requirements.txt`. For Ubuntu you have to install
system requirements first: read the comment in `proggers/requirements.txt`.
Consider using a virtualenv to encapsulate the dependencies.

Now to prepare the dataset:
  * Visit https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/ and
    download the data to some directory `CENSUS_DIR`. We only need
    `census-income.data.gz` (unpacked).
  * Create a postgres database and a user.
  * Copy the db setting examples and adjust them accordingly:
    * **NOTE:** adjusting the table name in the settings is not yet fully
      supported. Leave it as is.
    * `cp scripts/census/db_settings.example.py scripts/census/db_settings.py`
    * `vim scripts/census/db_settings.py`
    * `cp db_settings.example.py db_settings.py`
    * `vim db_settings.py`
  * Load the census dataset into the postgres database:
    `python scripts/census/census_loader.py CENSUS_DIR`

Then either run `./exp.sh` to run all experiments or explore the options of
`python experiment.py --help` to run a custom experiment setup.

Finally run `./exp_plot.sh OUT_DIR` to create the plots for the experiment
results. `OUT_DIR` is the directory where the results haven been placed it.
If `./exp.sh` was used, this will be `out_results`.


--------------------------------------------------------------------------------
\[1\] A. Galakatos, A. Crotty, E. Zgraggen, C. Binnig, and T. Kraska. Revisiting
reuse for approximate query processing. Proceedings of the VLDB Endowment,
10(10):1142–1153, 2017.
