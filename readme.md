scruples
========
A corpus and code for understanding norms and subjectivity.

This repository contains the code for building and analyzing the
`scruples` corpus. `scruples` is a corpus for studying _norm
understanding_ in anecdotes.

To rebuild the dataset or re-run the analyses, see [Setup](#setup) and
then [Quickstart](#quickstart). For documentation on how we validated
the extractions, see the [Annotation Guidelines](./docs/annotation-guidelines.md).

**Note: This repository is intended for research purpooses only.** It is
NOT intended for use in production environments, and there is no
intention for ongoing maintainence. See the [Disclaimer](#disclaimer)
for more information.


Setup
-----
This project requires Python 3.7. To setup the project:

  1. Make sure you have the MySQL client (on ubuntu):

         sudo apt-get install libmysqlclient-dev

  2. Install [PyTorch][pytorch] using the directions on their site.
  3. Install [apex][apex]:

         pip install \
           --no-cache-dir \
           --global-option="--cpp_ext" \
           --global-option="--cuda_ext" \
           git+git://github.com/nvidia/apex.git@453eefa56454142f8fc788478ad511973cc0fe1b

  4. Install this repository:

         pip install --editable .

  5. Download the english model for spacy:

         python -m spacy download en

  6. (optional) Run the tests to make sure that everything is
     working. They'll take about 5 minutes to complete:

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
      evaluate  Evaluate models on scruples.
      make      Make different components of scruples.

To build the dataset, you'll need to download the reddit
[posts][reddit-posts] and [comments][reddit-comments]. The initial
version of `scruples` used November 2018 through April 2019.


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


[apex]: https://github.com/nvidia/apex
[pytorch]: https://pytorch.org/
[reddit-comments]: http://files.pushshift.io/reddit/comments/
[reddit-posts]: http://files.pushshift.io/reddit/submissions/
