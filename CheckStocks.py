import glob
import pandas as pd
from pathlib import Path

# crete Main df
dfCheck = pd.DataFrame(columns=['fname', 'maxPopul'])

# check for popular shares
path = "popularity_export/*.csv"
for idx, pathName in enumerate(glob.glob(path)):
    df = pd.read_csv(pathName)
    xlist = df.users_holding.tolist()
    dfCheck.at[idx, 'fname'] = pathName
    dfCheck.at[idx, 'maxPopul'] = max(xlist)
    if idx == 10000:  # for testing
        break
dfCheck = dfCheck.sort_values(by=['maxPopul'], ascending=False)
dfCheck = dfCheck.iloc[1:round(len(dfCheck)*0.01)]  # Take only most popular (top 1%)

filepath = Path('usedStocks.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
dfCheck.to_csv('usedStocks.csv', index=False)
print('done')
