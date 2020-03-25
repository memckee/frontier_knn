import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


class EffFrontier():

    def __init__(self, port_history, exp_returns, risk_free, exp_freq='A'):
        self.history = port_history
        self.freq = exp_freq
        self.covars = self.history.cov()
        self.exp_returns = exp_returns
        self.rf = risk_free
        if self.freq == 'M':
            self.multiplier = 12
        else:
            self.multiplier = 1
        self.return_min = self.exp_returns.values.min() * self.multiplier
        self.return_max = self.exp_returns.values.max() * self.multiplier
        self.return_range = np.linspace(self.return_min, self.return_max, 200)
        self.frontier = self.generate_frontier()

    def port_return(self, weights):
        return np.sum(self.exp_returns*self.multiplier*weights)

    def port_vol(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.covars*12, weights)))

    def check_weights(self, weights):
        return np.sum(weights)-1

    def generate_frontier(self):
        vol_list = []
        weight_list = []
        init_guess = [1/len(self.history.columns)
                      for port in self.history.columns]
        bounds = [(0, 1) for port in self.history.columns]
        for possible_return in self.return_range:
            cons = ({'type': 'eq', 'fun': self.check_weights},
                    {'type': 'eq', 'fun': lambda w: self.port_return(w) - possible_return})
            result = minimize(self.port_vol, init_guess,
                              method='SLSQP', bounds=bounds, constraints=cons)
            vol_list.append(result['fun'])
            weight_list.append(result['x'])
        df = pd.DataFrame(data={'return': self.return_range,
                                'volatility': vol_list, 'weights': weight_list})
        return df

    def plot_frontier(self, add_random=False):
        plt.figure(figsize=(12, 8))
        if add_random == True:
            self.plot_random_weights()
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.plot(self.frontier['volatility'],
                 self.frontier['return'], 'b--', linewidth=3)
        plt.show()

    def plot_random_weights(self):
        np.random.seed(43)
        num_ports = 6000
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)
        rf_rate = self.get_rf()
        for x in range(num_ports):
            weights = np.array(np.random.random(len(self.history.columns)))
            weights = weights/np.sum(weights)
            ret_arr[x] = np.sum((self.exp_returns * self.multiplier * weights))
            vol_arr[x] = np.sqrt(
                np.dot(weights.T, np.dot(self.covars*12, weights)))
            sharpe_arr[x] = (ret_arr[x]-rf_rate)/vol_arr[x]
        plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap=sns.cubehelix_palette(
            start=2.8, rot=.1, as_cmap=True, reverse=True))
        plt.colorbar(label='Sharpe Ratio')

    def get_rf(self):
        rf = self.rf[self.rf.index.isin(self.history.index)]
        if self.freq == 'M':
            rf_rate = rf.mean()[0]
        else:
            rf_rate = (100*(((1+rf/100).prod())**(12/len(rf))-1))[0]
        return rf_rate

    def generate_five_levels(self):
        df = self.frontier.copy()
        df = df[df['return'] >= df.loc[df['volatility'].idxmin()]['return']]
        if len(df) == 0:
            return pd.concat([self.frontier.tail(1)]*5, ignore_index=True)
        elif len(df) < 5:
            return pd.concat([self.frontier.tail(len(df)),
                              pd.concat([self.frontier.tail(1)]*(5-len(df)), ignore_index=True)], ignore_index=True)
        else:
            df = df[df['volatility'].isin(df['volatility'].quantile(
                [0, .25, .50, .75, .90], interpolation='higher'))]
            return df


class PortStats():

    def __init__(self, portfolio, benchmark, risk_free):
        self.port = portfolio
        self.bm = benchmark
        self.rf = risk_free
        self.comb = pd.concat(
            [self.port, self.bm, self.rf], axis=1, join='inner')
        self.comb.columns = ['port', 'bm', 'rf']

    def cum_return_per(self, x):
        return (100*(np.prod(1+x/100)-1))

    def calc_ann_return(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        months = len(port)
        ann_return = 100*(((1+self.cum_return_per(port)/100)**(12/months))-1)
        return ann_return

    def rolling_ann_return(self, years=3):
        returns = self.port.rolling(
            years*12).apply(self.calc_ann_return, raw=False)
        return returns

    def calc_beta(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.bm], axis=1, join='inner')
        combi.columns = ['port', 'bm']
        return (combi.cov().loc['port', 'bm']/combi['bm'].var())

    def rolling_beta(self, years=3):
        beta = self.port.rolling(years*12).apply(self.calc_beta, raw=False)
        return beta

    def calc_alpha(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.bm, self.rf], axis=1, join='inner')
        combi.columns = ['port', 'bm', 'rf']
        beta = combi.cov().loc['port', 'bm']/combi['bm'].var()
        monthly_alpha = combi['port'] - combi['rf'] - \
            (beta * (combi['bm']-combi['rf']))
        return monthly_alpha.mean()*12

    def rolling_alpha(self, years=3):
        alpha = self.port.rolling(years*12).apply(self.calc_alpha, raw=False)
        return alpha

    def calc_vol(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        return port.std()*np.sqrt(12)

    def rolling_vol(self, years=3):
        vol = self.port.rolling(years*12).apply(self.calc_vol, raw=False)
        return vol

    def calc_sharpe(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.rf], axis=1, join='inner')
        combi.columns = ['port', 'rf']
        rp = self.calc_ann_return(port=combi['port'])
        rf = self.calc_ann_return(port=combi['rf'])
        port_vol = self.calc_vol(port=combi['port'])
        return (rp-rf)/port_vol

    def rolling_sharpe(self, years=3):
        sharpe = self.port.rolling(years*12).apply(self.calc_sharpe, raw=False)
        return sharpe

    def calc_treynor(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.rf], axis=1, join='inner')
        combi.columns = ['port', 'rf']
        rp = self.calc_ann_return(port=combi['port'])
        rf = self.calc_ann_return(port=combi['rf'])
        port_beta = self.calc_beta(port=combi['port'])
        if port_beta < 0.01:
            return np.NaN
        else:
            return (rp-rf)/port_beta

    def rolling_treynor(self, years=3):
        treynor = self.port.rolling(
            years*12).apply(self.calc_treynor, raw=False)
        return treynor

    def calc_sortino(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.rf], axis=1, join='inner')
        combi.columns = ['port', 'rf']
        downside_returns = combi[combi['port'] < combi['rf']]
        if len(downside_returns) == 0 or len(downside_returns) == 1:
            return np.NaN
        downside_deviation = np.sqrt(((downside_returns['port']-downside_returns['rf']).pow(
            2).sum())/(len(downside_returns)))*np.sqrt(12)
        rp = self.calc_ann_return(port=combi['port'])
        rf = self.calc_ann_return(port=combi['rf'])
        return (rp-rf)/downside_deviation

    def rolling_sortino(self, years=3):
        sortino = self.port.rolling(
            years*12).apply(self.calc_sortino, raw=False)
        return sortino

    def calc_up_cap(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.bm], axis=1, join='inner')
        combi.columns = ['port', 'bm']
        filtered = combi[combi['bm'] > 0]
        if filtered.empty:
            return np.NaN
        else:
            numerator = self.calc_ann_return(filtered['port'])
            denominator = self.calc_ann_return(filtered['bm'])
            return (numerator/denominator)*100

    def rolling_up_cap(self, years=3):
        up_cap = self.port.rolling(
            years*12).apply(self.calc_up_cap, raw=False)
        return up_cap

    def calc_down_cap(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        combi = pd.concat([port, self.bm], axis=1, join='inner')
        combi.columns = ['port', 'bm']
        filtered = combi[combi['bm'] < 0]
        if filtered.empty:
            return np.NaN
        else:
            numerator = self.calc_ann_return(filtered['port'])
            denominator = self.calc_ann_return(filtered['bm'])
            return (numerator/denominator)*100

    def rolling_down_cap(self, years=3):
        down_cap = self.port.rolling(
            years*12).apply(self.calc_down_cap, raw=False)
        return down_cap

    def calc_max_dd(self, port=None, years=None):
        if port is None:
            port = self.port
        if years != None:
            port = port.tail(years*12)
        first_row = {'date': port.index.min()-pd.offsets.MonthEnd(),
                     'return': 0.0}
        new_row = pd.DataFrame(
            data=[0.0], index=[port.index.min()-pd.offsets.MonthEnd()], columns=[0])
        mdd_returns = pd.concat([new_row, port], axis=0)
        mdd_returns['cum_returns'] = (1+(mdd_returns[0]/100)).cumprod()
        mdd_returns['cum_max'] = mdd_returns['cum_returns'].cummax()
        mdd_returns['drawdown'] = 1 - \
            mdd_returns['cum_returns'].div(mdd_returns['cum_max'])
        return mdd_returns['drawdown'].max()*100

    def rolling_max_dd(self, years=3):
        max_dd = self.port.rolling(
            years*12).apply(self.calc_max_dd, raw=False)
        return max_dd
