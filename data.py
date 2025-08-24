import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

pd.set_option("display.max_columns",None)

sp500=yf.Ticker('^GSPC')
sp500=sp500.history(period="max")
print(sp500.index)
sp500.plot.line(y="Close",use_index="True")
plt.show()         #Shows the graph of the stock

'''
del sp500["Dividends"]
del sp500["Stock Splits"]
'''

sp500['Tomorrow']=sp500['Close'].shift(-1)
target=sp500['Target']=(sp500['Tomorrow']>sp500['Close']).astype(int)
sp500=sp500.loc["1990-01-01":].copy()        #Copying the data from 1990-01-01 to the end

print(sp500)

model=RandomForestClassifier(n_estimators=300,min_samples_split=50,random_state=1)
train=sp500.iloc[:-100]     #Training the model on the data before 2023
test=sp500.iloc[-100:]      #Testing the model on the data after 2023
predictors=["Close","Volume","Open","High","Low"]
model.fit(train[predictors],train["Target"])


preds=model.predict(test[predictors])
preds=pd.Series(preds,index=test.index)
#print(precision_score(test["Target"],preds))

combined=pd.concat([test["Target"],preds],axis=1)
combined.plot()


def predict(train,test,predictors,model):
    model.fit(train[predictors],train['Target'])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds>=0.6]=1
    preds[preds<0.6]=0
    preds=pd.Series(preds,index=test.index, name="Predictions")
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined

def backset(data,model,predictors,start=2500,step=250):
    all_preds=[]
    for j in range(start,data.shape[0],step):
        train=data.iloc[0:j].copy()
        test=data.iloc[j:(j+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_preds.append(predictions)
    return pd.concat(all_preds)

horizons=[2,5,60,25,1000]
new_predictors=[]
for horizon in horizons:
    rolling_average=sp500.rolling(horizon).mean() #Calculate the rolling average of close price
    ratio_column=f"Close_Ratio_{horizon}"
    sp500[ratio_column]=sp500["Close"]/rolling_average["Close"]
    trend_column=f"Trend_{horizon}"
    sp500[trend_column]=sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors+=[ratio_column,trend_column]
    sp500=sp500.dropna()
    
print(sp500[new_predictors])
predictions=backset(sp500,model,new_predictors)
print(predictions["Predictions"].value_counts())  #Shows the number of times the model predicted 1(increase) and 0(decrease)
print(precision_score(predictions["Target"],predictions["Predictions"]))