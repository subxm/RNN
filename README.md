# One to One Character RNN

A minimal one to one recurrent neural network. Each prediction takes exactly one
character as input and produces exactly one character as output (timesteps = 1,
no sequence unrolling inside the model). Text generation is done by feeding the
predicted character back in as the next input, one step at a time, outside the model.

## Folder structure

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

## Setup (Windows / any OS)

1. Open a terminal in this folder.
2. Create a virtual environment (recommended but optional):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run

```
streamlit run app.py
```

This opens a browser tab. On the very first run, the app trains the model
automatically (takes under a minute on a laptop CPU with this dataset size) and
saves it to `model/`. Every run after that loads the saved model instantly.

If you'd rather train from the command line first:

```
python train_model.py
streamlit run app.py
```

## Using the app

1. Wait for "Model is ready".
2. Pick a starting character from the vocabulary list shown.
3. Choose how many characters to generate.
4. Click Generate.

## Notes

- The dataset in `dataset/corpus.txt` is small on purpose so training is fast.
  Replace it with a larger text file for more varied output; retrain by deleting
  the `model/` folder and rerunning.
- Architecture: `SimpleRNN(64) -> Dense(vocab_size, softmax)`, trained with
  categorical cross entropy.
