# Copyright Neal Fultz
# Apr 7 2024

from io import StringIO

import pandas as pd
import xgboost as xgb






class lpotato:

    def __init__(self, p_json, e_json, meta):
        """
        Params:
          app_json: path to p xgb json
          e_json:   path to e xgb json
          meta:     path to pandas dataframe for types.
        """

        self.model = xgb.XGBClassifier(enable_categorical=True)
        self.model.load_model(p_json)

        self.regmodel = xgb.XGBRegressor(enable_categorical=True)
        self.regmodel.load_model(e_json)

        self.meta = pd.read_pickle(meta)


    def clean(self, line):
        """
        Params:
        line:   line of data
        """
        df = pd.read_csv(StringIO(line), index_col=0, names=self.meta.columns, dtype=self.meta.dtypes.drop("date").to_dict(), header=None)
        return df


    def predict(self, df):
        """
        Params:
        df:     cleaned up data
        """

        X = df.drop(columns=['payout', 'date'])

        X['approvals'] = 1
        e_a1 = self.regmodel.predict(X)

        X['approvals'] = 0
        e_a0 = self.regmodel.predict(X)

        X = df.drop(columns=['payout','approvals', 'date'])

        p = self.model.predict_proba(X)[:, 1]

        E = e_a1 * p + e_a0 * (1-p)

        return E



    def score(self, line):
        """
        Params:
        line:   line of data
        """
        df = self.clean(line)
        score = self.predict(df)
        return score

if __name__ == "__main__":
    import sys
    my_model = lpotato(sys.argv[1], sys.argv[2], sys.argv[3])

    data = sys.stdin.readlines()
    print(my_model.score(data[0]))



