import streamlit as st
import CreateStockData as CSD
from LoadStockDataset import LoadStockDataset
from main import sklearn_tests as sk
import joblib

from datetime import datetime, timedelta
import pytz  # You might need to install this: pip install pytz

def get_pst_intervals(df):
    # Get the first element (assumed to be an int datetime)
    timestamp = df.iloc[0]
    
    # If timestamp is in milliseconds, convert to seconds
    if timestamp > 1e12:  # rough check
        timestamp = timestamp / 1000
    
    # Convert timestamp (assumed UTC) to datetime
    dt_utc = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
    
    # Convert UTC to PST (UTC-8)
    pst = pytz.timezone('America/Los_Angeles')
    dt_pst = dt_utc.astimezone(pst)
    output = dt_pst
    st.write(output)
    # Create list of next 5 intervals, 5 minutes apart, starting from dt_pst
    intervals = [(dt_pst + timedelta(minutes=5*i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(1, 6)]
    
    return intervals


def get_ticker(tickers,option):

    St =CSD.CreateStockData(1, 5, tickers, add_technical_indicators=True)
    if option == 1:
        St.get_stock_data()
        st.write("Stock Data Created for " + tick[0])
    if option == 2:
        data = St.get_recent_data()
        st.write("Stock Data refreshed for "  + tick[0])
        return data
    
def create_simple_model(tickers):
    ld = LoadStockDataset(tickers=tickers, dataset_index=1,normalize=0)
    dataset = ld.get_train_test_split()
    sk(id,dataset, tickers[0])
    st.write("Model created")

def predict_data(data,ticker):
    future_intervals = get_pst_intervals(data)
    obj = joblib.load(f'{ticker}_best_model.joblib')
    data = data.to_frame().T
    st.write(data.shape)
    predictions = obj.predict(data)
    # Flatten predictions
    st.write(future_intervals)
    st.write(predictions[:, 1:])


st.title("BTC Functions")
data = []

if st.button("Create new BTC Model"):
    tick = ["BTC-USD"]
    get_ticker(tick,1)
    create_simple_model(tick)

if st.button("Get Fresh BTC Data + Predict"):
    tick = ["BTC-USD"]
    predict_data(get_ticker(tick,2),tick[0])

st.title("ETH Functions")
if st.button("Create new ETH Model"):
    tick = ["ETH-USD"]
    st.session_state.data = get_ticker(tick,1)
    create_simple_model(tick)

if st.button("Get Fresh ETH Data + Predict"):
    tick = ["ETH-USD"]
    predict_data(get_ticker(tick,2),tick[0])

st.title("XRP Functions")
if st.button("Create new XRP Model"):
    tick = ["XRP-USD"]
    st.session_state.data = get_ticker(tick,1)
    create_simple_model(tick)

if st.button("Get Fresh XRP Data + Predict"):
    tick = ["XRP-USD"]
    predict_data(get_ticker(tick,2),tick[0])

st.title("SOL Functions")
if st.button("Create new SOL Model"):
    tick = ["SOL-USD"]
    st.session_state.data = get_ticker(tick,1)
    create_simple_model(tick)

if st.button("Get Fresh SOL Data + Predict"):
    tick = ["SOL-USD"]
    predict_data(get_ticker(tick,2),tick[0])

st.title("DOGE Functions")
if st.button("Create new DOGE Model"):
    tick = ["DOGE-USD"]
    st.session_state.data = get_ticker(tick,1)
    create_simple_model(tick)

if st.button("Get Fresh DOGE Data + Predict"):
    tick = ["DOGE-USD"]
    predict_data(get_ticker(tick,2),tick[0])

st.title("ADA Functions")
if st.button("Create new ADA Model"):
    tick = ["ADA-USD"]
    st.session_state.data = get_ticker(tick,1)
    create_simple_model(tick)

if st.button("Get Fresh ADA Data + Predict"):
    tick = ["ADA-USD"]
    predict_data(get_ticker(tick,2),tick[0])