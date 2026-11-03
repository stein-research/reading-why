# Data

**e-SNLI** (Camburu et al., NeurIPS 2018; https://github.com/OanaMariaCamburu/e-SNLI, MIT licence, see
[`LICENSE-e-SNLI`](LICENSE-e-SNLI)) extends **SNLI** (Bowman et al., EMNLP 2015; sentences under CC BY-SA 4.0) with
the words annotators highlighted and the reasons they wrote. The paper uses its four CSV files at commit
`7b585a3f`. They are not stored here: `src/esnli.py` (`fetch()`) downloads each file the first time a script or notebook
needs it into `data/esnli/` and checks it against [`SHA256SUMS`](SHA256SUMS); the demo fetches only the test split
(7 MB).

[`../results/esnli_vocab.json`](../results/esnli_vocab.json) is the models' vocabulary: `<pad>`, `<unk>`, `<sep>`
and the 22,316 lower-cased words seen at least three times in the training split (`src/esnli.py`). As a list of
words from the SNLI sentences it is shared under CC BY-SA 4.0.
