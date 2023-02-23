import pandas as pd
from datetime import time,date,datetime,timedelta
import requests


def get_data():
    import time 
    today_date = int(time.mktime(pd.Timestamp('2015-01-01').timetuple()))
    end_date = int(time.mktime((date.today() + pd.Timedelta('1D')).timetuple()))
    ticker = 'VN30F1M'
    link = "https://services.entrade.com.vn/chart-api/chart?from={start_date}&resolution=1&symbol={ticker}&to={end_date}".format(start_date=today_date, ticker=ticker,end_date=end_date)
    f = requests.get(link)
    dict_f = f.json()
    import datetime
    df = pd.DataFrame()
    df['Date'] = dict_f['t']
    df['Date'] = pd.to_datetime(df['Date'].astype(int).apply(lambda x: datetime.datetime.fromtimestamp(x)))
    df['Close'] = dict_f['c']
    df['High'] = dict_f['h']
    df['Low'] = dict_f['l']
    df['Open'] = dict_f['o']
    df['Volume'] = dict_f['v']
    df['day'] = df['Date'].dt.date
    df = df.sort_values('Date')
    return df

if __name__ == '__main__':
    name= 'vn30'
    data = get_data()
    data.to_excel('data_{}.xlsx'.format(name), index = False)