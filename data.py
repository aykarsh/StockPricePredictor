import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd


sp500=yf.Ticker('^GSPC')
sp500=sp500.history(period="max")
print(sp500.index)
sp500.plot.line(y="Close",use_index="True")
#plt.show()         Shows the graph of the stock
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500['Tomorrow']=sp500['Close'].shift(-1)
target=sp500['Target']=(sp500['Tomorrow']>sp500['Close']).astype(int)
sp500=sp500.loc["1990-01-01":].copy()
print(sp500)
model=RandomForestClassifier(n_estimators=1000,min_samples_split=100,random_state=1)
train=sp500.iloc[:-100]
test=sp500.iloc[-100:]

predictors=["Close","Volume","Open","High","Low"]
model.fit(train[predictors],train['Target'])
pred=model.predict(test[predictors])
pred=pd.Series(pred,index=test.index)
print(pred)
print(precision_score(test["Target"],pred))