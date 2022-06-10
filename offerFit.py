from __future__ import annotations
"""
py --version 3.81
pip install pandas
pip install numpy
pppp

"""

"""
Instructions:
def adjust_dtype(self, types: Dict[str, Any]) -> None:
Fill in the methods of the DataCleaner class to produce the same printed results
as in the comments below. Good luck, and have fun!
"""

import numpy as np
import pandas as pd
#from numpy import nan
#from numpy import datetime64

from typing import Dict, Any, List


class DataCleaner:
    """
    Transform a pandas df while keeping track of the history of transformations to
    allow reverting back to earlier state.
    """
    def __init__(self, df: pd.DataFrame):
      self.current=df
      self.history=[("Initial df",df.copy())]

    def adjust_dtype(self, types: Dict[str, Any]) -> None:
      desc = 'Adjust_dtype using ' + str(types)
      try:
        self.current['timestamp'] = pd.to_datetime(self.current['timestamp'])
        new_df = (desc,self.current.copy()) 
        self.history.append(new_df)
      except Exception as e: 
        self.log(desc,e)

    def impute_missing(self, columns: List[str]) -> None:
      desc = 'Impute_missing in '+str(columns)
      try:
        for column in columns:
          mean_column = self.current[column].mean()
          self.current[column].fillna(value=mean_column,inplace=True)
        new_df = (desc,self.current.copy()) 
        self.history.append(new_df)
      except Exception as e: 
        self.log(desc,e)    

    def revert(self, steps_back: int = 1) -> None:
      desc = 'Revert the last step(s) ' + str(steps_back)
      try:
        self.history=self.history[: -steps_back or None]
        self.current = self.history[-1][1]
      except Exception as e: 
        self.log(desc, e)  

    def save(self, path: str) -> None:
      self.current.to_pickle(path)

    @staticmethod
    def load(path: str):
      return DataCleaner(pd.read_pickle(path))

    def log(self, desc: str, e: Exception) -> None:
      print('*** Exception: ' + desc + ': ' + repr(e))

transactions = pd.DataFrame(
    {
        "customer_id": [10, 10, 13, 10, 11, 11, 10],
        "amount": [1.00, 1.31, 20.5, 0.5, 0.2, 0.2, np.nan],
        "timestamp": [
            "2020-10-08 11:32:01",
            "2020-10-08 13:45:00",
            "2020-10-07 05:10:30",
            "2020-10-08 12:30:00",
            "2020-10-07 01:29:33",
            "2020-10-08 13:45:00",
            "2020-10-09 02:05:21",
        ]
    }
)

transactions_dc = DataCleaner(transactions)

print(f"Current dataframe:\n{transactions_dc.current}")

# Current dataframe:
#    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21

print(f"Current dtypes:\n{transactions_dc.current.dtypes}")

# Initial dtypes:
# customer_id      int64
# amount         float64
# timestamp       object
# dtype: object

transactions_dc.adjust_dtype({"timestamp": np.datetime64}) 

print(f"Changed dtypes to:\n{transactions_dc.current.dtypes}")

# Changed dtypes to:
# customer_id             int64
# amount                float64
# timestamp      datetime64[ns]

transactions_dc.impute_missing(columns=["amount"])

print(f"Imputed missing as mean:\n{transactions_dc.current}")

# Imputed missing as mean:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

print(f"History of changes:\n{transactions_dc.history}")

# ** Any coherent structure with history of changes **
# E.g., here's one possibility

# History of changes:
# [('Initial df',    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21), ("Adjusted dtypes using {'timestamp': <class 'numpy.datetime64'>}",    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21), ("Imputed missing in ['amount']",    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21)]

transactions_dc.save("transactions")
loaded_dc = DataCleaner.load("transactions")
print(f"Loaded DataCleaner current df:\n{loaded_dc.current}")

# Loaded DataCleaner current df:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

transactions_dc.revert()
print(f"Reverting missing value imputation:\n{transactions_dc.current}")

# Reverting missing value imputation:
#    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21
