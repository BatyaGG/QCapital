import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")

class Calc_invest():
    
    def __init__(self, tick_df=None, fwd=5, commission=0.001, 
                 limit=0.5, spx_com=False):
        """
        Initializes a class that calculates return of a single trade. For example,
        if you want to buy Apple stock @$120 for $10000, it will calculate the number
        of whole shares you can buy, after commission (83 shares). The remaining cash
        is also accounted for. 
        Parameters:
            
            tick_df - dictionary with prices for given tickers (which are keys of the dict)
            fwd - number of days to hold a position for
            commission - $ value of trade commission =per= share
            limit - maximum value that a stock can rise (intends to limit overfit to rare explosive events)
            spx_com - whether to charge commission for trading SPX
        """
                
        self.tick_df = tick_df
        self.fwd = fwd
        self.commission = commission
        self.limit = limit
        self.spx_com = spx_com
    
    def calc_return(self, tick, date, portf_df, allocation=30000):
        """
        Calculates trade return for a given stock and date.
        Parameters:
            
            tick - ticker to buy
            date - date of position opening
            portf_df - dataframe with S&P 500 data, must include inXdays column,
                        which calculates SPX values in X number of days
            allocation - $ value of the trade
        
        Returns:
            
            end_value - $ value of the trade on exit
            
        """
        
        self.portf_df = portf_df
        
        if tick == 'spx':
            if self.spx_com:
                self.init_price = self.portf_df[tick].loc[date]
                self.end_price = self.portf_df['inXdays'].loc[date]
    
                self.stocks_bot = int(allocation/(self.init_price+self.commission))
                self.remainder = round(allocation - self.stocks_bot*(self.init_price+self.commission),2)
                self.end_value = round(self.remainder + self.stocks_bot*(self.end_price-self.commission), 2)
            else:
                self.init_price = self.portf_df[tick].loc[date]
                self.end_price = self.portf_df['inXdays'].loc[date]
    
                self.stocks_bot = int(allocation/self.init_price)
                self.remainder = round(allocation - self.stocks_bot*self.init_price,2)
                self.end_value = round(self.remainder + self.stocks_bot*self.end_price, 2)
            
            return self.end_value
        
        else:
            self.init_price = self.portf_df[tick].loc[date][tick]
            self.end_price = self.portf_df[tick].loc[date:].iloc[self.fwd][tick]

            if type(self.limit) == float:
                self.end_price = min( round(self.init_price*(1+self.limit), 2) , self.end_price )

            self.stocks_bot = int(allocation/(self.init_price+self.commission))
            self.remainder = round(allocation - self.stocks_bot*(self.init_price+self.commission),2)
            self.end_value = round(self.remainder + self.stocks_bot*(self.end_price-self.commission), 2)
            
            return self.end_value
        

class Portfolio():
    
    def __init__(self, pred, tick_df, portfolio=200000, days_shift=5, threshold=0.7, subparts=1, 
                  spx_dir='spx index.xlsx', upto_subparts=True, base_ticker='aa',
                  calc_invest=Calc_invest):
        """
        Calculates portfolio return over the 'pred' period.
            v 12.04.21
        Parameters:

            pred - pd.Series with date:tickers in the index and probabilities as values
            porfolio - value of portfolio. It affects the ability to buy full shares and thus should be large enough
            days_shift - forecasting period
            threshold - cut-off probability threshold
            subparts - maximum number of stocks to buy according to the model's recommendations
            spx_dir - directory of S&P 500 data
            upto_subparts - whether to mix forecasts with spx; must have subparts > 1 to work
                            For example, if there is only one forecasted ticker, but two can be
                            purchased in 1 day, 50% of purchase power will be allocated to spx
            base_ticker - reference ticker to base date limits on. Should be a company with that has
                            been public during the whole period

        Attributes:

            portf_df - pd.DataFrame with forecasted portfolio values
            
        """
        
        self.pred = pd.DataFrame(pred)
        self.pred.columns = ['prob']
        self.pred['datetick'] = self.pred.index
        self.pred['date'] = self.pred['datetick'].apply(lambda x: x.split(":")[0])
        self.pred['tick'] = self.pred['datetick'].apply(lambda x: x.split(":")[1])
        
        pred1 = self.pred[self.pred.prob >= threshold]
        idf = pred1.pivot_table(values='prob', index='date', columns='tick').dropna(how='all')

        self.portf_df = pd.read_excel(spx_dir, index_col='date')
        self.portf_df = self.portf_df.loc[pred1.date[0]:pred1.date[-1]]
        self.portf_df['inXdays'] = self.portf_df['spx'].shift(-days_shift)
        parts = days_shift
        self.portf_df['portfolio'] = np.nan
        self.portf_df['portfolio'].iloc[0] = portfolio
        self.portf_df['part'] = np.arange(0, self.portf_df.shape[0])
        self.portf_df['part'] = self.portf_df['part'] % parts + 1

        part_columns = []
        for part in range(1,1+parts):
            pcol = 'part'+str(part)
            self.portf_df[pcol] = self.portf_df['portfolio']/parts
            part_columns.append(pcol)

        iseries = pd.Series(self.portf_df.index).shift(-days_shift)
        iseries.index = self.portf_df.index

        chosen_tickers = []

        for date in tqdm_notebook(self.portf_df.iloc[:].index):
            date = str(date).split(' ')[0]
            ipart = 'part' + str(int(self.portf_df.loc[date].part))
            self.portf_df.loc[:date] = self.portf_df.loc[:date].fillna(method='ffill')


            if date in idf.index:

                tickers = idf.loc[date].dropna()
                tot_allocation = round(self.portf_df[ipart].loc[date], 2)

                if tickers.shape[0] > subparts:
                    tickers = tickers.sort_values(ascending=False).iloc[0:subparts]
                elif tickers.shape[0] < subparts:

                    if upto_subparts == False:
                        for s in range(subparts - tickers.shape[0]):
                            tickers = tickers.append(pd.Series({'spx':1}))
                    else:
                        subparts = tickers.shape[0]

                tot_end_value = []
                
                x=0
                actual_tickers = list(tickers.index)
                
                for tick in tickers.index:
                    
                    try:
                        c = calc_invest.calc_return(tick=tick, date=date, portf_df=tick_df, 
                                   allocation=round(tot_allocation/subparts, 2))
                    except:
                        c = calc_invest.calc_return(tick='spx', date=date, portf_df=self.portf_df,  
                                   allocation=round(tot_allocation/subparts, 2))
                        actual_tickers[x] = 'spx'
                        
                    x+=1 

                    tot_end_value.append(c)


                tot_end_value = round(np.sum(tot_end_value),2)
                try:
                    end_date = tick_df[base_ticker].loc[date:].index[days_shift]
                except:
                    end_date = iseries[date]

                self.portf_df[ipart].loc[end_date] = tot_end_value

                chosen_tickers.append(str(actual_tickers))

            else:

                tot_end_value = calc_invest.calc_return(tick='spx', date=date, portf_df=self.portf_df,  
                                   allocation=round(self.portf_df[ipart].loc[date], 2))

                #tot_end_value = c.end_value

                end_date = iseries[date]
                self.portf_df[ipart].loc[end_date] = tot_end_value
                chosen_tickers.append('spx')

        self.portf_df['chosen'] = chosen_tickers
        self.portf_df['portfolio'] = self.portf_df[part_columns].sum(axis=1)
        self.portf_df['spx100'] = self.portf_df.spx/self.portf_df.spx.iloc[0]*100
        self.portf_df['p100'] = self.portf_df.portfolio/self.portf_df.portfolio.iloc[0]*100



