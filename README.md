# RNN Projects

This repository contains all three RNN demos as sibling folders so they show
up side by side at the repo root:

- [OneToOneRNN](OneToOneRNN)
- [ManytoOneRNN](ManytoOneRNN)
- [ManytoManyRNN](ManytoManyRNN)

## ManytoOneRNN

A spam classification demo built with a many-to-one RNN. The model consumes a
sequence of words and produces one output label for the whole message.

### Folder structure

```
ManytoOneRNN/
  app.py              Streamlit UI
  requirements.txt
  spam.csv            Training dataset
  spam_model.keras    Saved model
  tokenizer.pkl       Saved tokenizer
```

### Run

Open a terminal inside `ManytoOneRNN` and run:

```
pip install -r requirements.txt
streamlit run app.py
```

If the app is started from the repo root, the paths still resolve correctly
because they are now based on the app file location.

## OneToOneRNN

A minimal one to one recurrent neural network. Each prediction takes exactly
one character as input and produces exactly one character as output
(timesteps = 1, no sequence unrolling inside the model). Text generation is
done by feeding the predicted character back in as the next input, one step at
a time, outside the model.

### Folder structure

```
OneToOneRNN/
  app.py              Streamlit UI
  train_model.py      Standalone training script (optional, app.py trains automatically too)
  requirements.txt
  dataset/
    corpus.txt        Training text
  model/              Created automatically after first training run
    model.keras
    mappings.pkl
```

### Run

Open a terminal inside `OneToOneRNN` and run:

```
pip install -r requirements.txt
streamlit run app.py
```

If you'd rather train from the command line first:

```
python train_model.py
streamlit run app.py
```

## ManytoManyRNN

The many-to-many project is stored in its own folder alongside the one-to-one
project. Open that folder directly to run its Streamlit app and use its saved
model and tokenizer files.

## Notes

- The one-to-one dataset in `OneToOneRNN/dataset/corpus.txt` is small on
  purpose so training is fast. Replace it with a larger text file for more
  varied output; retrain by deleting the `OneToOneRNN/model/` folder and
  rerunning.
- Architecture: `SimpleRNN(64) -> Dense(vocab_size, softmax)`, trained with
  categorical cross entropy.
