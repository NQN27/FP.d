import pandas as pd
import numpy as np
from datetime import time,date,datetime,timedelta
import requests
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

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



def backtest_position_ps(position, price):
    a = pd.DataFrame()
    pos = pd.Series(position)
    pr = pd.Series(price)
    pos_long = np.where(pos>0, pos, 0)
    pos_short = np.where(pos<0, pos, 0)
    pnl_long = (pr.shift(-1)-pr) * pos_long
    pnl_short = (pr.shift(-1)-pr) * pos_short
    fees = abs(pos.diff(1)).cummax()*0.037
    return pnl_long + pnl_short - fees
  
def stop_loss_profit(long):
    ### đầu vào sẽ là 1 DataFrame bao gồm những columns, vị trí như sau:
    ### ['Date','Close','pos_change','pos','Long_cut_loss','Short_cut_loss','Long_cut_profit','Short_cut_profit'] 
    ### Date: columns thòi gian theo minute
    ### Close: giá tại thời điểm đó
    ### pos_change: dự định ban đầu không có stop loss, thì vị thế biến đổi sẽ như thế nào
    ### pos: dự định ban đầu không có stop loss, thì vị thế dữ lại sẽ như thế nào(cumsum của cột pos_change)
    ### Long_cut_loss: giá cut loss, cột này sẽ hơi khó tạo, khi mà giá nhỏ hơn (không bao gồm bằng) mức giá, thì sẽ thoát hết lệnh
    ### Long_cut_profit: khi mà pos lớn hơn 0, nghĩa là đang giữ long, khi giá lớn hơn price threshold sẽ thoát lệnh để chốt lời
    ### Short_cut_loss: tương tử như cốt trên, nhưng mà sẽ là khi giá lớn hơn mực độ thì sẽ thoát 
    ### Short_cut_profit: khi mà pos nhỏ hơn 0, nghĩa là đang dữ short, khi giá nhỏ hơn 1 mức sẽ thoát lệnh để chốt lời
    test = long.copy()
    test['pos1'] = test['pos_change'].cumsum()
    test.loc[(test['pos1']!=test['pos1'].shift())&(test['pos1']!=0)&(test['pos1'].notna())&(test['pos1'].shift()==0),'enter_exit'] = 'enter'
    test.loc[(test.index!=test.index[0])&(test['pos1']!=test['pos1'].shift())&(test['pos1']==0)&(test['pos1'].notna())&(test['pos1'].shift()!=0),'enter_exit'] = 'exit'
    test = test.drop(['pos1'],axis=1)
    test.index = range(len(test))
    test = test.loc[test.index>=test.loc[(test['pos_change']!=0)&(test['pos_change'].notna())].sort_values('Date').head(1).index.values[0]]
    test = np.array(test)
    test_len = range(len(test))
    enter_exit = []
    for i in tqdm(test_len):
        if i==0:
            test[i,3] = test[i,2]
            enter_exit.append('enter')
            li = i
        else:
            if (test[i,2]==0):
                test[i,3] = test[li,3]
                li = i
            if (len(enter_exit)>0)&(test[li,3]>=1)&(test[i,1]<test[i,4])&(enter_exit[-1]=='enter'):
                test[i,2] = test[li,3]*-1
                test[i,3] = 0
                test[i,8] = 'exit'
                enter_exit.append('exit')
                li = i
            elif (len(enter_exit)>0)&(test[li,3]>=1)&(test[i,1]>test[i,6])&(enter_exit[-1]=='enter'):
                test[i,2] = test[li,3]*-1
                test[i,3] = 0
                test[i,8] = 'exit'
                enter_exit.append('exit')
                li = i
            elif (len(enter_exit)>0)&(test[li,3]<=-1)&(test[i,1]>test[i,5])&(enter_exit[-1]=='enter'):
                test[i,2] = test[li,3]*-1
                test[i,3] = 0
                test[i,8] = 'exit'
                enter_exit.append('exit')
                li = i
            elif (len(enter_exit)>0)&(test[li,3]<=-1)&(test[i,1]<test[i,7])&(enter_exit[-1]=='enter'):
                test[i,2] = test[li,3]*-1
                test[i,3] = 0
                test[i,8] = 'exit'
                enter_exit.append('exit')
                li = i
            else:
                if (len(enter_exit)!=0)&(enter_exit[-1]=='exit')&(test[i,2]!=0)&(test[i,8]=='enter'):
                    test[i,3] = test[li,3] + test[i,2]
                    test[i,8] = 'enter'
                    enter_exit.append('enter')
                    li = i
                elif (len(enter_exit)!=0)&(enter_exit[-1]=='enter')&(test[i,2]!=0)&(test[i,8]=='exit'):
                    test[i,3] = test[li,3] + test[i,2]
                    test[i,8] = 'exit'
                    enter_exit.append('exit')
                    li = i
                elif (len(enter_exit)!=0)&(enter_exit[-1]=='enter')&(test[i,2]!=0)&(test[i,8]=='enter'):
                    test[i,2] = 0
                    test[i,3] = test[li,3] + test[i,2]
                    test[i,8] = np.nan
                    li = i
                elif (len(enter_exit)!=0)&(enter_exit[-1]=='exit')&(test[i,2]!=0)&(test[i,8]=='exit'):
                    test[i,2] = 0
                    test[i,3] = test[li,3] + test[i,2]
                    test[i,8] = np.nan
                    li = i                    
    test_ = pd.DataFrame(test)
    test_.columns = ['Date','Close','pos_change','pos','Long_cut_loss','Short_cut_loss','Long_cut_profit','Short_cut_profit','enter_exit']
    return test_,enter_exit
  
def sharpe(pnl):
    return (pnl.mean()/pnl.std() * np.sqrt(252))

def max_drawdown(pnl,pnl_column_name,pnl_price_column_name):
    ## pnl_column_name: ten columns cua pnl
    ## pnl_price_column_name: ten columns cua gia
    return ((pnl[pnl_column_name].cumsum() - pnl[pnl_column_name].cumsum().cummax()).min())/pnl[pnl_price_column_name].max()

def back_test_infomation(pnl,pnl_column_name,pnl_price_column_name):
    s = pnl.reset_index().groupby(pd.to_datetime(pnl.reset_index()['day']).dt.year).apply(lambda x:sharpe(x[pnl_column_name].cumsum())).to_frame().rename(columns={0:'sharpe_ratio'})
    r = pnl.reset_index().groupby(pd.to_datetime(pnl.reset_index()['day']).dt.year).apply(lambda x:x[pnl_column_name].cumsum().iloc[-1]).to_frame().rename(columns={0:'return_point'})
    m = pnl.reset_index().groupby(pd.to_datetime(pnl.reset_index()['day']).dt.year).apply(lambda x:max_drawdown(x,pnl_column_name,'Close')).to_frame().rename(columns={0:'max_drawdown'})
    shar = sharpe(pnl[pnl_column_name].cumsum())
    ret = pnl[pnl_column_name].cumsum().iloc[-1]
    md = max_drawdown(pnl,pnl_column_name,pnl_price_column_name)
    total_year = pd.DataFrame(np.array([ret,shar,md]),columns=['Total_Year'],index=['return_point','sharpe_ratio','max_drawdown']).T
    return pd.concat([r.merge(s,right_index=True,left_index=True,how='left').merge(m,right_index=True,left_index=True,how='left'),total_year])
  
# def back_test_infomation(pnl,pnl_column_name,pnl_price_column_name):
#     s = pnl.reset_index().groupby(pd.to_datetime(pnl.reset_index()['day']).dt.year).apply(lambda x:sharpe_1(x[pnl_column_name].cumsum())).to_frame().rename(columns={0:'sharpe_ratio'})
#     r = pnl.reset_index().groupby(pd.to_datetime(pnl.reset_index()['day']).dt.year).apply(lambda x:x[pnl_column_name].cumsum().iloc[-1]).to_frame().rename(columns={0:'return_point'})
#     m = pnl.reset_index().groupby(pd.to_datetime(pnl.reset_index()['day']).dt.year).apply(lambda x:max_drawdown(x,pnl_column_name,'Close')).to_frame().rename(columns={0:'max_drawdown'})
#     sha = sharpe_1(pnl[pnl_column_name].cumsum())
#     ret = pnl[pnl_column_name].cumsum().iloc[-1]
#     md = max_drawdown(pnl,pnl_column_name,pnl_price_column_name)
#     total_year = pd.DataFrame(np.array([ret,sha,md]),columns=['Total_Year'],index=['return_point','sharpe_ratio','max_drawdown']).T
#     return pd.concat([r.merge(s,right_index=True,left_index=True,how='left').merge(m,right_index=True,left_index=True,how='left'),total_year])
  
def pnl_year_plot(pnl,pnl_column_name):
    sns.set()
    pnl['year'] = pd.to_datetime(pnl['day']).dt.year
    uni_year = pnl['year'].unique().tolist()
    import math
    if len(uni_year)+1<=3:
        plot_x_len = 1
        plot_y_len = len(uni_year)+1
    else:
        plot_x_len = math.ceil((len(uni_year)+1)/3)
        plot_y_len = 3
    fig, axes = plt.subplots(plot_x_len, plot_y_len, figsize=(12,8), sharey=True)
    for i,year in enumerate(uni_year):
        x_pnl = pnl.loc[pnl['year']==year]['day'].values
        y_pnl = pnl.loc[pnl['year']==year][pnl_column_name].cumsum().values
        sns.lineplot(ax=axes[math.floor((i+1-0.01)/plot_y_len),i-(math.floor((i+1-0.01)/plot_y_len)*3)], x=x_pnl, y=y_pnl)
    i = len(uni_year)
    x_pnl = pnl['day'].values
    y_pnl = pnl['profit'].cumsum().values
    sns.lineplot(ax=axes[math.floor((i+1-0.01)/plot_y_len),i-(math.floor((i+1-0.01)/plot_y_len)*3)], x=x_pnl, y=y_pnl)
if __name__ == '__main__':
    data = get_data()
    data.to_csv('data.csv',index=False)