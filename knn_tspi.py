import pandas as pd
import numpy as np
from numba import jit
from sklearn.neighbors import KNeighborsRegressor


class TimeSeriesKNN():

    def __init__(self, time_series, metric=None, k=5, length=5, step=1):
        self.ts = time_series
        self.k = k
        self.length = length
        self.step = step
        if metric is None:
            self.model = KNeighborsRegressor(algorithm='brute')
        else:
            self.model = KNeighborsRegressor(
                algorithm='brute', metric=self.cid)

    @staticmethod
    @jit
    def cid(x, y):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        x_diff = np.diff(x)
        y_diff = np.diff(y)
        ce_x = np.sqrt(x_diff.dot(x_diff))
        ce_y = np.sqrt(y_diff.dot(y_diff))
        ces = np.asarray([ce_x, ce_y])
        try:
            ce = np.max(ces)/np.min(ces)
        except:
            ce = np.max(ces)
        return np.linalg.norm(x-y) * ce

    def normalize_series(self, length=None, step_ahead=None, time_series=None):
        if length is None:
            length = self.length
        if step_ahead is None:
            step_ahead = self.step
        if time_series is None:
            time_series = self.ts
        sub_cols = ['ts_' + str(l) for l in range(length)]
        x_cols = ['z_' + str(l) for l in range(length)]
        df = pd.concat([time_series.shift(l).rename(sub_cols[l])
                        for l in range(length)], axis=1)
        df['actual'] = df[sub_cols[0]].shift(-step_ahead)
        df['mean'] = df[sub_cols].mean(axis=1)
        df['std'] = df[sub_cols].std(axis=1)
        df = pd.concat([df, pd.concat([df[sub_cols[l]].subtract(df['mean']).div(
            df['std']).rename(x_cols[l]) for l in range(length)], axis=1)], axis=1)
        df['actual_z'] = df['actual'].subtract(df['mean']).div(df['std'])
        y_col = 'actual_z'
        df = df.iloc[length-1:]
        df = df[df['std'] != 0]
        return df, x_cols, y_col

    def predict(self, model, train_y, test_x, x_mean, x_std, k):
        locs = model.kneighbors(np.reshape(
            test_x.to_numpy(), (1, -1)), n_neighbors=k, return_distance=False)
        neighbors = pd.concat([train_y.iloc[row] for row in locs])
        prediction = np.mean((neighbors*x_std + x_mean))
        return prediction

    def predict_history(self, start=1000, step=None, length=None, k=None):
        if step is None:
            step = self.step
        if length is None:
            length = self.length
        if k is None:
            k = self.k
        model = self.model
        ml_set, x_cols, y_col = self.normalize_series(length, step)
        prediction_list = []

        for i in range(len(ml_set)-start-step+1):
            # print('Test {} of {}'.format(i, (len(ml_set)-start)))
            train = ml_set.iloc[:min([start+i-step+1, len(ml_set)-step])]
            test = ml_set.iloc[start+i+step-1]
            model.fit(train[x_cols], train[y_col])
            prediction_list.append(
                self.predict(model, train[y_col], test[x_cols], test['mean'], test['std'], k))
        pred_series = pd.Series(data=prediction_list,
                                index=ml_set.iloc[start+step-1:].index)
        values = pd.concat([ml_set, pred_series.rename(
            'prediction')], axis=1, join='outer')
        values = values[['ts_0', 'actual', 'prediction']].shift(step)
        values = values.rename(columns={'ts_0': 'actual_prev'})
        values['actual_return'] = 100 * \
            (values['actual'].div(values['actual_prev'])-1)
        values['predicted_return'] = 100 * \
            (values['prediction'].div(values['actual_prev'])-1)
        return values.dropna(axis=0, how='any')

    def predict_one(self, step=None, length=None, k=None):
        if length is None:
            length = self.length
        if k is None:
            k = self.k
        if step is None:
            step = self.step

        model = self.model

        ml_set, x_cols, y_col = self.normalize_series(
            length=length, step_ahead=step, time_series=self.ts)
        train = ml_set.iloc[:len(ml_set)-step]
        test = ml_set.iloc[-1]
        model.fit(train[x_cols], train[y_col])
        prediction = self.predict(
            model, train[y_col], test[x_cols], test['mean'], test['std'], k)
        return prediction

    def predict_next(self, date=None, horizon=None, length=None, k=None):
        if length is None:
            length = self.length
        if k is None:
            k = self.k
        if horizon is None:
            horizon = 90
        if date is None:
            date = self.ts.index[-91]
            index = self.ts.index[-90:]
        elif pd.to_datetime(date) >= self.ts.index[-horizon]:
            index = pd.bdate_range(
                start=date + pd.tseries.offsets.BDay(1), periods=horizon)
        elif pd.to_datetime(date) < self.ts.index[-horizon]:
            date = self.ts.index[self.ts.index >= date].min()
            index = self.ts.index[self.ts.index > date][:horizon]

        model = self.model

        prediction_list = []
        history = self.ts[self.ts.index <= date]

        for i in range(horizon):
            ml_set, x_cols, y_col = self.normalize_series(
                length=length, step_ahead=i+1, time_series=history)
            train = ml_set.iloc[:len(ml_set)-i-1]
            test = ml_set.iloc[-1]
            model.fit(train[x_cols], train[y_col])
            prediction_list.append(self.predict(
                model, train[y_col], test[x_cols], test['mean'], test['std'], k))
        pred_series = pd.Series(data=prediction_list,
                                index=index, name='prediction')
        values = pd.merge(pred_series, self.ts.rename(
            'actual'), how='left', left_index=True, right_index=True)

        return values[['actual', 'prediction']]

    def get_metrics(self, values):
        metrics = {}
        values['error'] = values['actual']-values['prediction']
        values['error_return'] = values['actual_return'] - \
            values['predicted_return']
        values['naive'] = values['actual']-values['actual_prev']
        metrics['mse'] = values['error'].pow(2).mean()
        metrics['tu'] = (values['error'].pow(
            2).sum())/(values['naive'].pow(2).sum())
        metrics['mase'] = (values['error'].abs().mean()) / \
            (values['naive'].abs().mean())
        metrics['mape'] = (values['error']/values['actual']).abs().mean()*100
        metrics['rmse_return'] = np.sqrt(values['error_return'].pow(2).mean())
        metrics['nrmse_return'] = metrics['rmse_return'] / \
            (np.abs(values['actual_return'].abs().mean()))

        tp = len(values[(values['actual_return'] > 0)
                        & (values['predicted_return'] > 0)])
        tn = len(values[(values['actual_return'] < 0)
                        & (values['predicted_return'] < 0)])
        fp = len(values[(values['actual_return'] < 0)
                        & (values['predicted_return'] > 0)])
        fn = len(values[(values['actual_return'] > 0)
                        & (values['predicted_return'] < 0)])

        metrics['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
        metrics['precision'] = tp/(tp+fp)
        metrics['recall'] = tp/(tp+fn)
        metrics['fscore'] = (2*metrics['precision']*metrics['recall']
                             ) / (metrics['precision']+metrics['recall'])

        return metrics

    def evaluate_model(self, start=None, step=None, length=None, k=None):
        if start is None:
            start = round(len(self.ts)*.5)
        if step is None:
            step = self.step
        if length is None:
            length = self.length
        if k is None:
            k = self.k
        metrics = self.get_metrics(self.predict_history(
            start=start, step=step, length=length, k=k))

        return metrics
