Scruples
========
A corpus and code for understanding norms and subjectivity.

This repository is for the paper: [Scruples: A Corpus of Community Ethical
Judgments on 32,000 Real-life Anecdotes][paper]. In particular, `scruples` is a
collection of datasets for studying _norm understanding_ in anecdotes. This
repo contains code for building and analyzing `scruples`, running the
baselines, and demoing the models and BEST estimator.

To download the data, see [Data](#data).

To rebuild the dataset or re-run the analyses, see [Setup](#setup) and
then [Quickstart](#quickstart). For documentation on how we validated
the extractions, see the [Annotation Guidelines](./docs/annotation-guidelines.md).

To cite the paper, jump to [Citation](#citation).

**Note: This repository is intended for research purpooses only.** It is
NOT intended for use in production environments, and there is no
intention for ongoing maintainence. See the [Disclaimer](#disclaimer)
for more information.


Data
----
`scruples` has two primary datasets: the Anecdotes and the Dilemmas.

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
Once you've [installed the package](#setup), you'll have the `scruples`
CLI available. It's a hierarchical, self-documenting CLI that contains
all the commands necessary to build and analyze `scruples`:

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
version of `scruples` used November 2018 through April 2019.

Also, `scruples` comes with demos that you can run and view locally in the
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

`scruples` encourages progress on this research problem by providing a
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
[paper]: https://arxiv.org/abs/2008.09094
[pytorch]: https://pytorch.org/
[reddit-comments]: http://files.pushshift.io/reddit/comments/
[reddit-posts]: http://files.pushshift.io/reddit/submissions/
