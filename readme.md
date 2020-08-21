Scruples
========
A corpus and code for understanding norms and subjectivity.

This repository is for the paper: [Scruples: A Corpus of Community Ethical
Judgments on 32,000 Real-life Anecdotes][paper]. Scruples provides datasets for
studying _norm understanding_ in anecdotes and language. This repo contains
code for building and analyzing the Scruples datasets, running the baselines,
demoing the models and the BEST estimator, and using the BEST estimator
directly to estimate the best possible performance on classification datasets.

Jump to a section of this readme to accomplish different goals:

  - [Data](#data): Download the Scruples data.
  - [Demos](#demos): View or run demos for the BEST estimator or the model
    trained to predict people's ethical judgments on anecdotes and moral
    dilemmas.
  - [Setup](#setup): Install the code in this repository.
  - [Quickstart](#quickstart): Get started running the code in this repo.
  - [Citation](#citation): Cite the Scruples project.
  - [Contact](#contact): Reach out with questions or comments.
  - [Disclaimer](#disclaimer): Understand the intended purpose of this work as
    well as it's limitations.

In addition, the following documents dive deep into particular topics:

  - [Annotation Guidelines](./docs/annotation-guidelines.md): Learn how we
    annotated data for Scruples to validate the extraction performance.
  - [Demos](./docs/demos.md): Set up and run the demos on your own machine.

**Note: This repository is intended for research purpooses only.** It is
NOT intended for use in production environments, and there is no
intention for ongoing maintainence. See the [Disclaimer](#disclaimer)
for more information.


Data
----
Scruples has two primary datasets: the Anecdotes and the Dilemmas.

### Anecdotes

The Anecdotes provide 32,000 anecdotes of real-life situations with
ethical judgments collected from community members about who was in the
wrong. See [Scruples: A Corpus of Community Ethical Judgments on
32,000 Real-life Anecdotes][paper] for more information.

You can download the Anecdotes [here][anecdotes].

### Dilemmas

The Scruples Dilemmas provide 10,000 ethical dilemmas in the form of
paired actions, where the model must identify which one was considered
less ethical by crowd workers on Mechanical Turk. See [Scruples: A
Corpus of Community Ethical Judgments on 32,000 Real-life
Anecdotes][paper] for more information.

You can download the Dilemmas [here][dilemmas].


Demos
-----
Scruples has two demos associated with it.

### Scoracle

Visit [scoracle][scoracle] to compute the BEST (Bayesian Estimated Score
Terminus) performance for a classification dataset. BEST uses the annotations
to estimate the upper bound for how well models can possibly do on a dataset
under various metrics (accuracy, cross entropy, etc.). See [the paper][paper]
for details.

### Norms

The [norms][norms] demo shows how current neural models can learn to predict
basic ethical judgments using the Scruples data. It let's you run anecdotes and
dilemmas through a model to view its predictions. In addition, it visualizes
how Dirichlet-multinomial layers allow models to separate intrinsic from model
uncertainty. [The paper][paper] elaborates on these techniques.

### Running the Demos

Running the demos yourself is quite easy! If you want to run these demos on
your own hardware, check out the [demo documentation](./docs/demos.md).


Setup
-----
This project requires Python 3.7. To setup the project:

  1. Make sure you have the MySQL client (on ubuntu):

         sudo apt-get install libmysqlclient-dev

  2. Install [PyTorch][pytorch] using the directions on their site.
  3. Install this repository:

         pip install --editable .

  4. Download the english model for spacy:

         python -m spacy download en

  5. (optional) Run the tests to make sure that everything is
     working. They'll take about 5 minutes to complete, or you can pass the
     `--skip-slow` (`-s`) option to run a smaller, faster test suite:

         pip install pytest
         pytest


Quickstart
----------
Once you've [installed the package](#setup), you'll have the Scruples
CLI available. It's a hierarchical, self-documenting CLI that contains
all the commands necessary to build and analyze Scruples:

    $ scruples --help
    Usage: scruples [OPTIONS] COMMAND [ARGS]...

      The command line interface for scruples.

    Options:
      --verbose  Set the log level to DEBUG.
      --help     Show this message and exit.

    Commands:
      analyze   Run an analysis.
      demo      Run a demo's server.
      evaluate  Evaluate models on scruples.
      make      Make different components of scruples.

To build the dataset, you'll need to download the reddit
[posts][reddit-posts] and [comments][reddit-comments]. The initial
version of Scruples used November 2018 through April 2019.

Also, Scruples comes with demos that you can run and view locally in the
browser:

    $ scruples demo --help
    Usage: scruples demo [OPTIONS] COMMAND [ARGS]...

      Run a demo's server.

    Options:
      --help  Show this message and exit.

    Commands:
      norms     Serve the norms demo.
      scoracle  Serve the scoracle demo.


Citation
--------
If you build off of this code, data, or work, please cite [the paper][paper] as
follows:

    @article{Lourie2020Scruples,
        author = {Nicholas Lourie and Ronan Le Bras and Yejin Choi},
        title = {Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes},
        journal = {arXiv e-prints},
        year = {2020},
        archivePrefix = {arXiv},
        eprint = {2008.09094},
    }


Contact
-------
For public, non-sensitive questions and concerns, please file an issue
on this repository.

For private or sensitive inquiries email mosaic on the allenai.org
website.


Disclaimer
----------
**This code and corpus is intended for research purposes only**.

As AI agents become more autonomous, they must understand and apply
human ethics, values, and norms. One step towards better _norm
understanding_ is to reproduce normative judgments drawn from various
communities. This skill would enable computers to anticipate people's
reactions and understand deeper situational context.

Scruples encourages progress on this research problem by providing a
corpus of real-world ethical situations with community sourced normative
judgments. The norms expressed by this corpus represent those of the
community from which they're drawn&mdash;and thus they are not
necessarily the right norms to use in any particular application
scenario.

Any organization looking to incorporate normative understanding into a
product or service should carefully consider, investigate, and evaluate
which norms are correct for their particular application.


[anecdotes]: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz
[dilemmas]: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/dilemmas.tar.gz
[norms]: https://norms.apps.allenai.org/
[paper]: https://arxiv.org/abs/2008.09094
[pytorch]: https://pytorch.org/
[reddit-comments]: http://files.pushshift.io/reddit/comments/
[reddit-posts]: http://files.pushshift.io/reddit/submissions/
[scoracle]: https://scoracle.apps.allenai.org/
