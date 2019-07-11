Annotation Guidelines
=====================
Guidelines for annotating the ground truth for the data.

These guidelines define how to annotate the ground truth for the reddit
data in order to produce extraction quality evaluations. There are two
kinds of annotations used in evaluating the extractions: comment and
post annotations.


Comment Annotations
-------------------
Comment annotations should be written in a JSON Lines file where each
object has the following keys:

<dl>
  <dt><code>id</code></dt>
  <dd>
    the ID attribute of the corresponding comment
  </dd>
  <dt><code>label</code></dt>
  <dd>
    a gold annotation for the label (one of <code>"AUTHOR"</code>,
    <code>"OTHER"</code>, <code>"EVERYBODY"</code>,
    <code>"NOBODY"</code>, or <code>"INFO"</code>) expressed by the
    comment, or <code>null</code> if no label is expressed
  </dd>
  <dt><code>implied</code></dt>
  <dd>
    <code>true</code> if the label is implied by the view of the author
    and <code>false</code> if the label is somehow explicitly stated
  </dd>
  <dt><code>spam</code></dt>
  <dd>
    <code>true</code> if the comment is spam, <code>false</code>
    otherwise
  </dd>
</dl>

The possible labels are:

<dl>
  <dt><code>AUTHOR</code></dt>
  <dd>
    The author of the anecdote is in the wrong.
  </dd>
  <dt><code>OTHER</code></dt>
  <dd>
    The other person in the anecdote is in the wrong.
  </dd>
  <dt><code>EVERYBODY</code></dt>
  <dd>
    Everyone in the anecdote is in the wrong.
  </dd>
  <dt><code>NOBODY</code></dt>
  <dd>
    No one in the anecdote is in the wrong.
  </dd>
  <dt><code>INFO</code></dt>
  <dd>
    More information is required to make a judgment.
  </dd>
</dl>

If the comment explicitly expresses a label either by it's initialism or
some phrase corresponding to the initialism, then use that label for the
comment. Similarly, mark the comment with `implied` as `false` and
`spam` as `false`.

If the comment expresses multiple labels or is otherwise ambiguous, mark
`label` as `null`, `implied` as `null`, and `spam` as `true`.

If the comment expresses no labels explicitly but still has a viewpoint
that clearly expresses one of the labels, then use that label for the
comment. Mark `implied` as `true` and `spam` as `false`.

Finally, if the comment expresses no label (i.e., none of `AUTHOR`,
`OTHER`, `NOBODY`, `EVERYBODY`, or `INFO`), then mark `label` as `null`,
`implied` as `null`, and `spam` as `true`.


Post Annotations
----------------
Post annotations should be written in a JSON Lines file where each
object has the following keys:

<dl>
  <dt><code>id</code></dt>
  <dd>
    the ID attribute of the corresponding post
  </dd>
  <dt><code>post_type</code></dt>
  <dd>
    a gold annotation for the post's type
  </dd>
  <dt><code>implied</code></dt>
  <dd>
    <code>true</code> if the post type is not explicitly stated in the
    post title.
  </dd>
  <dt><code>spam</code></dt>
  <dd>
    <code>true</code> if the post is spam, <code>false</code> otherwise
  </dd>
</dl>

Possible post types are:

<dl>
  <dt><code>HISTORICAL</code></dt>
  <dd>The author is asking "am I the a&ast;&ast;hole?"</dd>
  <dt><code>HYPOTHETICAL</code></dt>
  <dd>The author is asking "would I be the a&ast;&ast;hole?"</dd>
  <dt><code>META</code></dt>
  <dd>The post is about the subreddit itself.</dd>
</dl>

If the post type is explicitly stated in the post title, then mark
`post_type` as the stated post type, mark `implied` as `false`, and
`spam` as `false`, unless the post type is `META` in which case mark
spam as `true`. Additionally, if the post type is explicitly stated but
clearly wrong (such as using HISTORICAL for a HYPOTHETICAL post), then
use the true post type rather than the stated one.

If the post type is not explicitly stated in the post title, but
otherwise clear from the post, mark the appropriate post type, mark
`implied` as `true` and `spam` as `false`.

If the post cannot be categorized into one of the types above, mark the
`post_type` as `null`, `implied` as `null`, and `spam` as `true`.

If the post is something that should not be present in the dataset (for
example a deleted post), then mark `spam` as `true`.
