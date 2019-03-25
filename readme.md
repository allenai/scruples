socialnorms
===========
A benchmark for detecting social norm violations in anecdotes.

This repository contains code for creating the `socialnorms` dataset: a
benchmark for detecting social norm violations in anecdotes.

For more information about the project, read the
[Background](#background) section. If you're just looking to rebuild the
dataset, skip to [Setup](#setup) and then to [Quickstart](#quickstart).


Background
----------
This project creates a dataset for detecting social norm violations,
using data from the [Am I the A&ast;&ast;hole][amitheasshole-subreddit]
subreddit.

### Overview: The `socialnorms` Dataset

Each instance in the dataset has some title text, body text, and then a
label which is one of:

  1. **YTA**  : The author of the anecdote is in the wrong.
  2. **NTA**  : The other person in the anecdote is in the wrong.
  3. **ESH**  : Everyone in the anecdote is in the wrong.
  4. **NAH**  : No one in the anecdote is in the wrong.
  5. **INFO** : More information is required to make a judgment.

In addition, instances have a unique id as well as scores associated
with each of the possible labels.

Here's an example of what a single instance might look like:

    {
      "id": "a4z0e9",
      "title": "AITA for hosing a Spider down a Plughole?",
      "text": "I panicked,  am I going to hell where spiders get their revenge?",
      "label_scores": {
        "YTA": 2,
        "NTA": 1,
        "ESH": 0,
        "NAH": 0,
        "INFO": 0
      },
      "label": "YTA"
    }

Though usually, the `"text"` attribute is significantly longer.

### Data Source

The [Am I the A&ast;&ast;hole][amitheasshole-subreddit] subreddit is an
online community where people can ask whether things they have done or
intend to do would violate social or ethical norms.

Usually, people post anecdotes drawn from their own experiences and ask
whether or not they (or the other person involved) handled the situation
well. Members of the community then comment with their opinion, usually
by leaving one of the acronyms discussed in
[Overview](#overview-the-socialnorms-dataset).

Posts talking about an anecdote begin with `AITA` (Am I the
A&ast;&ast;hole) and posts discussing hypothetical situations begin with
`WIBTA` (Would I Be the A&ast;&ast;hole).

### Dataset Details

In building the dataset, title and body text come directly from the
reddit posts. Body text is extracted from the comments where a bot
automatically records the original version of the posts -- so usually
the text for each instance should represent the post before any later
edits.

The label scores can loosely be interpreted as being proportional to the
probability that a member of that subreddit would agree with that
label. To produce label scores, each top-level comment is assigned an
automatically extracted label using regular expressions. Every comment
counts as one vote towards the final score of the label. Comments with
multiple extracted labels, i.e. ambiguous comments, are dropped from the
calculation. The label with the highest score becomes the overall label
for that instance.

Only top-level comments are considered when creating the label scores
because those comments are directly responding to the post (rather than
other comments).


Setup
-----
This project requires Python 3.7. To install the requirements:

```
pip install -r requirements.txt
```


Quickstart
----------
To build the dataset, first download the reddit [posts][reddit-posts]
and [comments][reddit-comments]. Then, run
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
