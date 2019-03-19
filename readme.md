socialnorms
===========
A benchmark for detecting social norm violations in anecdotes.

This repository contains code for creating the `socialnorms` dataset: a
benchmark for detecting social norm violations in anecdotes.

For more information about the project, read the [Background](#background)
section. If you're just looking to rebuild the dataset, skip to
[Setup](#setup) and then to [Quickstart](#quickstart).


Background
----------
This project creates a dataset for detecting social norm violations,
using data from the [Am I the A**hole][amitheasshole-subreddit]
subreddit.

Each instance in the dataset has some title text, body text, and then a
label which is one of:

  1. **YTA**  : The author of the anecdote is in the wrong.
  2. **NTA**  : The other person in the anecdote is in the wrong.
  3. **ESH**  : Everyone in the anecdote is in the wrong.
  4. **NAH**  : No one in the anecdote is in the wrong.
  5. **INFO** : More information is required to make a judgment.

In addition, instances have a unique id as well as scores associated
with each of the possible labels.

Title and body text come directly from the reddit posts. To produce
label scores, each comment is assigned a label using regular
expressions, and then the score on each comment is summed into a total
score for that label. Ambiguous comments are dropped. The label with the
highest score becomes the overall label for that instance.


Setup
-----
This project requires Python 3.7. To install the requirements:

```
pip install -r requirements.txt
```


Quickstart
----------
To build the dataset, first download the reddit
[posts][reddit-posts] and [comments][reddit-comments]. Then, run
[`bin/make-dataset.py`][make-dataset.py] with the proper inputs to
create the dataset. `bin/make-dataset.py` is self-documenting:

```
$ python bin/make-dataset.py --help
Usage: make-dataset.py [OPTIONS] COMMENTS_PATH POSTS_PATH OUTPUT_PATH

  Create the social norms dataset and write it to OUTPUT_PATH.

  Read in the reddit posts from POSTS_PATH and comments from COMMENTS_PATH,
  create the social norms dataset, and write it to OUTPUT_PATH.

Options:
  --help  Show this message and exit.
```


[amitheasshole-subreddit]: https://www.reddit.com/r/AmItheAsshole/
[make-dataset.py]: ./bin/make-dataset.py
[reddit-posts]: http://files.pushshift.io/reddit/submissions/
[reddit-comments]: http://files.pushshift.io/reddit/comments/
