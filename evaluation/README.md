# Evaluation

In our paper, we tested the different decoding methods with respect to the following dimensions:

1. Quality (BLEU4, CIDEr, SPICE):
  - https://github.com/Maluuba/nlg-eval
2. Diversity (TTR, number of word types, % novel, % coverage):
  - follow installation instructions & run `batch_evaluate.py` in submodule `MeasureDiversity`
3. Informativity (Recall @ 1):
  - follow installation instructions & run `batch_evaluate.py` in submodule `vsepp`
4. Further Metrics (average frequency ranks for types and tokens, frequency distribution over POS tags, distance from WordNet root):
  - lexical_analysis.py
  - pos_wordnet.py
