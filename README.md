# lpotato payout model

(c) Neal Fultz, Apr 7 2024

* Generate models and features using included notebook.
* slides in `docs/`

To install:
```
pip3 install git+https://github.com/nfultz/lpotato
```

To run:
```
python3 -m lpotato p_xgb.json e_xgb.json df.pkl <line.txt
```

TODO:
* Move training out of notebook into module.
* Scale up hyperparameter search.
* Model pruning.
* Remove scikit-learn, use xgb directly.
