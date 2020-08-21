Demos
=====
Scruples ships with two demos, [scoracle](#scoracle) and [norms](#norms). You
can visit these demos live on the web, or you can run them yourself.


Setup
-----
To run either of the demos, first you need to follow the
[Setup](../readme.md#setup) and [Quickstart](../readme.md#quickstart)
documentation in the [readme](../readme.md).


Scoracle
--------
To run scoracle, use the `scruples demo scoracle` command. The command is
self-documenting:

    $ scruples demo scoracle --help
    Usage: scruples demo scoracle [OPTIONS]

      Serve the scoracle demo.

    Options:
      --expose        Whether to expose the server to the internet, or run on
                      localhost.
      --port INTEGER  The port on which to serve the demo. Defaults to 5000.
      --help          Show this message and exit.

So, to run scoracle on localhost at port 5000, execute:

    scruples demo scoracle

When you visit the site, you should see something like this:

![Scoracle Demo About Screenshot](./assets/demo-screenshots/scoracle-about.png?raw=true "Scoracle Demo About")

After submitting a dataset and requesting the BEST performance for some
metrics, you'll see something like this:

![Scoracle Demo Results Screenshot](./assets/demo-screenshots/scoracle-results.png?raw=true "Scoracle Demo Results")


Norms
-----
Before running the `norms` demo, you'll have to download the config files and
weights:

  - **Scruples Anecdotes demo model**: [config][anecdotes-demo-config]
    [weights][anecdotes-demo-weights]
  - **Scruples Dilemmas demo model**: [config][dilemmas-demo-config]
    [weights][dilemmas-demo-weights]

Once you've obtained the weights, set the following environment variables:

    SCRUPLES_NORMS_ACTIONS_BASELINE=roberta
    SCRUPLES_NORMS_ACTIONS_MODEL=$DILEMMAS_MODEL_PATH
    SCRUPLES_NORMS_CORPUS_BASELINE=roberta
    SCRUPLES_NORMS_CORPUS_MODEL=$ANECDOTES_MODEL_PATH
    SCRUPLES_NORMS_PREDICT_BATCH_SIZE=$BATCH_SIZE
    SCRUPLES_NORMS_GPU_IDS=$GPU_IDS

For each of the models, the path to it should be a directory containing both
the config and the weights file. `$GPU_IDS` should be a comma separated string
of integers corresponding to which GPUs to use. You can set
`SCRUPLES_NORMS_GPU_IDS` to the empty string to use the CPU instead. Start out
with a value of 1 or 2 for `$BATCH_SIZE`, increasing it if you need more
performance and have hardware that can support larger batches.

To run norms, use the `scruples demo norms` command. The command is
self-documenting:

    $ scruples demo norms --help
    Usage: scruples demo norms [OPTIONS]

      Serve the norms demo.

      In order to run this server, you must set the following environment
      variables:

          SCRUPLES_NORMS_ACTIONS_BASELINE   : The baseline to use for the resource
            (actions).
          SCRUPLES_NORMS_ACTIONS_MODEL      : The path to the saved pretrained
            model to use for predicting the actions.
          SCRUPLES_NORMS_CORPUS_BASELINE    : The baseline to use for the corpus.
          SCRUPLES_NORMS_CORPUS_MODEL       : The path to the saved pretrained
            model to use for predicting the corpus.
          SCRUPLES_NORMS_PREDICT_BATCH_SIZE : The batch size to use for
            prediction.
          SCRUPLES_NORMS_GPU_IDS            : A comma separated list of GPU IDs to
            use. If none are provided, then the CPU will be used instead.

    Options:
      --expose        Whether to expose the server to the internet, or run on
                      localhost.
      --port INTEGER  The port on which to serve the demo. Defaults to 5050.
      --help          Show this message and exit.

So, to run norms on localhost at port 5050, execute:

    scruples demo norms

When you visit the site, you should see something like this:

![Norms Demo About Screenshot](./assets/demo-screenshots/norms-about.png?raw=true "Norms Demo About")

And after submitting a dilemma to the model, you should see results like this:

![Norms Demo Results Screenshot](./assets/demo-screenshots/norms-results.png?raw=true "Norms Demo Results")


[anecdotes-demo-config]: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/corpus/config.json
[anecdotes-demo-weights]: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/corpus/pytorch_model.bin
[dilemmas-demo-config]: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/actions/config.json
[dilemmas-demo-weights]: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/actions/pytorch_model.bin
