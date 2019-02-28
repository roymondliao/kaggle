# contains the trainer logic that manages the job.

from __future__ import print_function
import keras
import tensorflow as tf
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
import time
import os
import gc
import traceback
import subprocess
import trainer.model as models
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io


class KaggleCompeitions:
    local_data_path = Path('/tmp')
    features_data_name = 'feature_data_3000.csv'
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']# , 'minute', 'second']
    split_probability = 0.1
    seed = 202109
    full_data = True
    def __init__(self, _data_source, _output_file, _output_path, _train_rows, _nchunk, _total_rows, _model_name,
                 _is_debug, _is_prepare_data):
        self.data_source_path = _data_source
        self.output_file = 'submission' + '_' + _output_file
        self.output_path = _output_path
        self.train_rows = _train_rows
        self.nchunk = _nchunk
        self.total_rows = _total_rows
        self.mn = _model_name
        self.is_debug = _is_debug
        self.is_prepare_data = _is_prepare_data

    def copy_data_from_gcs(self):
        """
        1. The data source include train data and test data.
        2. Data format use csv format.
        """
        cmd = 'gsutil -m cp {cloud} {local}'.format(cloud=os.path.join(self.data_source_path,'*.csv'),
                                                                       local=str(self.local_data_path))

        res = subprocess.check_call(cmd, shell=True)
        if not res:
            print('[INFO] Copy data successes...')
        else:
            print('[ERROR] Copy data failed, please check...')

    def load_data(self):
        dtypes = {
            'ip': 'uint32',
            'app': 'uint16',
            'device': 'uint16',
            'os': 'uint16',
            'channel': 'uint16',
            'is_attributed': 'uint8',
            'click_id': 'uint32'
        }

        if self.is_debug:
            test_data = pd.read_csv(str(self.local_data_path / 'test.csv'),
                                    dtype=dtypes, parse_dates=['click_time'],
                                    nrows=100000,
                                    usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
        else:
            print('Loading test data...')
            test_data = pd.read_csv(str(self.local_data_path / 'test.csv'),
                                    dtype=dtypes, parse_dates=['click_time'],
                                    usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

        if self.full_data:
            print('Loading full train data...')
            train_data = pd.read_csv(str(self.local_data_path / 'train.csv'),
                                     nrows=self.train_rows,
                                     dtype=dtypes, parse_dates=['click_time'],
                                     usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
        else:
            print('Loading train data...')
            train_data = pd.read_csv(str(self.local_data_path / 'train.csv'),
                                     skiprows=range(1, self.total_rows-self.nchunk), nrows=self.train_rows,
                                     dtype=dtypes, parse_dates=['click_time'],
                                     usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

        #self.train_data = pd.read_csv(str(self.local_data_path / 'feature_data.csv'))
        #print(self.train_data.columns)
        #self.length_of_train = self.train_rows
        self.length_of_train = len(train_data)
        self.train_data = train_data.append(test_data)
        del test_data
        gc.collect()

    def load_feature_data(self):
        cmd = 'gsutil -m cp {cloud} {local}'.format(cloud=os.path.join(self.output_path, self.features_data_name),
                                                    local=str(self.local_data_path))
        res = subprocess.check_call(cmd, shell=True)
        if not res:
            print('[INFO] Copy feature data: {} succeed...'.format(self.features_data_name))
        else:
            print('[ERROR] Copy feature data failed, please check...')
        self.train_data = pd.read_csv(str(self.local_data_path / self.features_data_name))
        self.length_of_train = self.train_rows

    @staticmethod
    def time_labels(t):
        most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
        least_freq_hours_in_test_data = [6, 11, 15]
        if t in most_freq_hours_in_test_data:
            res = 1
        elif t in least_freq_hours_in_test_data:
            res = 3
        else:
            res = 2
        return res

    @staticmethod
    def do_count(data, group_cols, target_col, new_col_name, col_type):
        print('[INFO] Count {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
        cols = group_cols.copy()
        cols.append(target_col)
        group_data = data[cols].groupby(by=group_cols)[[target_col]].count().reset_index().\
            rename(index=str, columns={target_col: new_col_name})
        data = data.merge(group_data, on=group_cols, how='left')
        del group_data
        data[new_col_name] = data[new_col_name].astype(col_type)
        gc.collect()
        return data

    @staticmethod
    def do_count_unique(data, group_cols, target_col, new_col_name, col_type):
        print('[INFO] Count unique {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
        cols = group_cols.copy()
        cols.append(target_col)
        group_data = data[cols].groupby(by=group_cols)[[target_col]].nunique().reset_index().\
            rename(index=str, columns={target_col: new_col_name})
        data = data.merge(group_data, on=group_cols, how='left')
        del group_data
        data[new_col_name] = data[new_col_name].astype(col_type)
        gc.collect()
        return data

    @staticmethod
    def do_cumcount(data, group_cols, target_col, new_col_name, col_type):
        print('[INFO] Cumcount {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
        cols = group_cols.copy()
        cols.append(target_col)
        group_data = data[cols].groupby(by=group_cols)[[target_col]].cumcount()
        data[new_col_name] = group_data.values
        del group_data
        data[new_col_name] = data[new_col_name].astype(col_type)
        gc.collect()
        return data

    @staticmethod
    def do_mean(data, group_cols, target_col, new_col_name, col_type):
        print('[INFO] Compute mean {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
        cols = group_cols.copy()
        cols.append(target_col)
        group_data = data[cols].groupby(by=group_cols)[[target_col]].mean().reset_index().\
            rename(index=str, columns={target_col: new_col_name})
        data = data.merge(group_data, on=group_cols, how='left')
        del group_data
        # data[new_col_name] = data[new_col_name].astype(col_type)
        gc.collect()
        return data

    @staticmethod
    def do_var(data, group_cols, target_col, new_col_name, col_type):
        print('[INFO] Compute var {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
        cols = group_cols.copy()
        cols.append(target_col)
        group_data = data[cols].groupby(by=group_cols)[[target_col]].var().reset_index().\
            rename(index=str, columns={target_col: new_col_name})
        data = data.merge(group_data, on=group_cols, how='left')
        del group_data
        # data[new_col_name] = data[new_col_name].astype(col_type)
        gc.collect()
        return data

    @staticmethod
    def next_click(data):
        # next_click: 定義user在不同時間點不同的channel中看到相同的app，click_buffer保留最後一個click的時間
        # The log2-transformed and rounded counts are further hashed with the feature name. So the original feature
        # gets mapped into separate features in the log-scale bins. This improves the score considerably and allows
        # modeling a non-linear response for each feature. Factors further model interactions between the
        # non-linear, although I haven't tested how much this contributes.
        D= 2**26
        data['category'] = (data['ip'].astype(str) + "_" + data['app'].astype(str) + "_" +
                            data['device'].astype(str) + "_" + data['os'].astype(str)).apply(hash) % D
        # 預先initial一個非常大的時間
        # Click_buffer maintains the last seen times of clicks for each category. For each click, you look if there's
        # been a previous click, and compute the time difference. This is done in reverse, and reversing the output
        # list next_clicks at the end gives the times of next clicks in the correct order.
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)
        data['epochtime'] = data['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, time in zip(reversed(data['category'].values), reversed(data['epochtime'].values)):
            next_clicks.append(click_buffer[category]-time)
            click_buffer[category]= time
        del(click_buffer)
        data.drop(['epochtime'], axis=1, inplace=True)
        data['next_click'] = list(reversed(next_clicks))
        return data

    @staticmethod
    def previous_click(data):
        D = 2**26
        data['category'] = (data['ip'].astype(str) + "_" + data['app'].astype(str) + "_" +
                            data['device'].astype(str) + "_" + data['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)
        data['epochtime'] = data['click_time'].astype(np.int64) // 10 ** 9
        prev_clicks= []
        for category, time in zip(data['category'].values, data['epochtime'].values):
            prev_clicks.append(time-click_buffer[category])
            click_buffer[category]= time
        del(click_buffer)
        data.drop(['epochtime'], axis=1, inplace=True)
        data['prev_click'] = prev_clicks
        return data

    @staticmethod
    def do_next_click(data, group_cols, col_type):
        print('[INFO] Compute next click with group by {} combination...'.format('-'.join(group_cols)))
        # Calculate the time to next click for each group
        # Name of new feature
        new_feature = '{}_nextClick'.format('_'.join(group_cols))

        # Unique list of features to select
        all_features = group_cols + ['click_time']
        data[new_feature] = (data[all_features].groupby(group_cols).click_time.shift(-1) -
                             data.click_time).dt.seconds.astype(col_type)
        gc.collect()
        return data

    @staticmethod
    def do_prev_click(data, group_cols, col_type):
        print('[INFO] Compute next click with group by {} combination...'.format('-'.join(group_cols)))
        # Calculate the time to next click for each group
        # Name of new feature
        new_feature = '{}_prevClick'.format('_'.join(group_cols))

        # Unique list of features to select
        all_features = group_cols + ['click_time']
        data[new_feature] = (data.click_time - data[all_features].groupby(group_cols).click_time.shift(+1)).\
            dt.seconds.astype(col_type)
        gc.collect()
        return data

    def features_engineering(self, features_version):
        # time features
        self.train_data['day'] = self.train_data['click_time'].dt.day.astype('uint8')
        self.train_data['hour'] = self.train_data['click_time'].dt.hour.astype('uint8')
        self.train_data['minute'] = self.train_data['click_time'].dt.minute.astype('uint8')
        self.train_data['second'] = self.train_data['click_time'].dt.second.astype('uint8')
        gc.collect()

        # Version - 1
        if features_version == 1:
            self.train_data['in_test_hh'] = self.train_data['hour'].apply(self.time_labels)
            self.train_data['in_test_hh'] = self.train_data['in_test_hh'].astype('uint8')
            # count channel with group ip-app-device-os
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'app', 'device', 'os'],
                                            target_col='channel', new_col_name='UsrappCount', col_type='uint16')
            self.train_data['UsrappNewness'] = self.train_data.groupby(['ip', 'app', 'device', 'os']).cumcount() + 1
            self.train_data['UsrappNewness'] = self.train_data['UsrappNewness'].astype('uint16')

            # count channel with group ip-device-os
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'device', 'os'],
                                            target_col='channel', new_col_name='UsrCount', col_type='uint16')
            self.train_data['UsrNewness'] = self.train_data.groupby(['ip', 'device', 'os']).cumcount() + 1
            self.train_data['UsrNewness'] = self.train_data['UsrNewness'].astype('uint16')
            self.train_data['UsrNewness'] = self.train_data['UsrNewness'].astype('uint16')

            # count channel with group ip-day-in_test_hh
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'day', 'in_test_hh'],
                                            target_col='channel', new_col_name='n_ip_day_test_hh', col_type='uint32')

            # count channel with group ip-day-hour
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'day', 'hour'],
                                            target_col='channel', new_col_name='n_ip', col_type='uint16')

            # count channel with group ip-day-hour-os
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'day', 'hour', 'os'],
                                            target_col='channel', new_col_name='n_ip_os', col_type='uint32')


            # count channel with group ip-day-hour-app
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'day', 'hour', 'app'],
                                            target_col='channel', new_col_name='n_ip_app', col_type='uint16')


            # count channel with group ip-day-hour-app-os
            self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'day', 'hour', 'app', 'os'],
                                            target_col='channel', new_col_name='n_ip_app_os', col_type='uint16')

            # count channel with group app-day-hour
            self.train_data = self.do_count(data=self.train_data, group_cols=['app', 'day', 'hour'],
                                            target_col='channel', new_col_name='n_app', col_type='uint16')

            # Computer ration of channel
            self.train_data = self.do_count(data=self.train_data, group_cols=['device', 'channel', 'hour'],
                                            target_col='app', new_col_name='n_dev_channel', col_type='uint32')
            self.train_data = self.do_count(data=self.train_data, group_cols=['device', 'channel', 'hour', 'app'],
                                            target_col='os', new_col_name='n_dev_channel_app', col_type='uint32')
            self.train_data['app_confRate'] = self.train_data['n_dev_channel_app']/self.train_data['n_dev_channel']

            self.train_data = self.next_click(data=self.train_data)
            self.train_data = self.previous_click(data=self.train_data)

            self.train_data.drop(['day', 'click_time', 'n_dev_channel', 'n_dev_channel_app'],
                                 axis=1, inplace=True)

        elif features_version == 2:
            # Version - 2
            try:
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['ip'],
                                                       target_col='channel', new_col_name='uni_ip_with_ch', col_type='uint32')
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['ip', 'device', 'os'],
                                                       target_col='app', new_col_name='uni_ip_dev_os', col_type='uint32')
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['ip', 'day'],
                                                       target_col='hour', new_col_name='uni_ip_day', col_type='uint32')
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['ip'],
                                                       target_col='app', new_col_name='uni_ip_with_app', col_type='uint32')
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['ip', 'app'],
                                                       target_col='os', new_col_name='uni_ip_app', col_type='uint32')
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['ip'],
                                                       target_col='device', new_col_name='uni_ip_with_dev', col_type='uint32')
                self.train_data = self.do_count_unique(data=self.train_data, group_cols=['app'],
                                                       target_col='channel', new_col_name='uni_app_with_ch', col_type='uint32')

                self.train_data = self.do_cumcount(data=self.train_data, group_cols=['ip'],
                                                   target_col='os', new_col_name='cum_ip', col_type='uint32')
                self.train_data = self.do_cumcount(data=self.train_data, group_cols=['ip', 'device', 'os'],
                                                   target_col='app', new_col_name='cum_ip_dev_os', col_type='uint32')

                self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'day', 'hour'],
                                                   target_col='channel', new_col_name='count_ip_day_hour', col_type='uint32')
                self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'app'],
                                                   target_col='channel', new_col_name='count_app', col_type='uint32')
                self.train_data = self.do_count(data=self.train_data, group_cols=['ip', 'app', 'os'],
                                                   target_col='channel', new_col_name='count_ip_app_os', col_type='uint32')

                #self.train_data = self.do_var(data=self.train_data, group_cols=['ip', 'day', 'channel'],
                #                              target_col='hour', new_col_name='var_ip_day_ch', col_type='float32')
                self.train_data = self.do_var(data=self.train_data, group_cols=['ip', 'app', 'os'],
                                              target_col='hour', new_col_name='var_ip_app_os', col_type='float32')
                #self.train_data = self.do_var(data=self.train_data, group_cols=['ip', 'app', 'channel'],
                #                              target_col='day', new_col_name='var_ip_app_ch', col_type='float32')

                self.train_data = self.do_mean(data=self.train_data, group_cols=['ip', 'app', 'channel'],
                                              target_col='hour', new_col_name='mean_ip_app_ch', col_type='float32')

                self.train_data = self.do_next_click(data=self.train_data,
                                                     group_cols=['ip', 'app', 'device', 'os', 'channel'],
                                                     col_type='float32')
                self.train_data = self.do_next_click(data=self.train_data,
                                                     group_cols=['ip', 'os', 'device'],
                                                     col_type='float32')
                self.train_data = self.do_next_click(data=self.train_data,
                                                     group_cols=['ip', 'os', 'device', 'app'],
                                                     col_type='float32')
                self.train_data = self.do_next_click(data=self.train_data,
                                                     group_cols=['device', 'channel'],
                                                     col_type='float32')
                self.train_data = self.do_next_click(data=self.train_data,
                                                     group_cols=['app', 'device', 'channel'],
                                                     col_type='float32')
                self.train_data = self.do_prev_click(data=self.train_data,
                                                     group_cols=['ip', 'channel'],
                                                     col_type='float32')
                self.train_data = self.do_prev_click(data=self.train_data,
                                                     group_cols=['ip', 'os'],
                                                     col_type='float32')

                # Computer ration of channel
                self.train_data = self.do_count(data=self.train_data, group_cols=['device', 'channel', 'hour'],
                                                target_col='app', new_col_name='n_dev_channel', col_type='uint32')
                self.train_data = self.do_count(data=self.train_data, group_cols=['device', 'channel', 'hour', 'app'],
                                                target_col='os', new_col_name='n_dev_channel_app', col_type='uint32')
                self.train_data['app_confRate'] = self.train_data['n_dev_channel_app']/self.train_data['n_dev_channel']

                #self.train_data = self.next_click(data=self.train_data)
                #self.train_data = self.previous_click(data=self.train_data)
            except Exception as e:
                print(e, traceback.print_exc(()))

            # remove useless features
            print('[INFO] Remove useless features')
            self.train_data.drop(['ip', 'day', 'click_time', 'n_dev_channel', 'n_dev_channel_app'],
                                 axis=1, inplace=True)
            gc.collect()

        self.train_data.to_csv(str(self.local_data_path / 'feature_data_full.csv'), index=False)
        cmd = 'gsutil cp {} {}'.format(str(self.local_data_path / 'feature_data_full.csv'), self.output_path)
        subprocess.check_call(cmd, shell=True)
        gc.collect()

    def split_data(self):
        self.test_df = self.train_data[self.length_of_train:]
        self.train_df, self.validation_df = train_test_split(self.train_data[:self.length_of_train],
                                                             test_size=self.split_probability, random_state=self.seed)
        print("train size: ", len(self.train_df))
        print("valid size: ", len(self.validation_df))
        print("test size : ", len(self.test_df))

    def train_model(self, model_name):
        categorical_features = [col for col in self.train_df.columns if col in self.categorical]
        self.predictors = list(self.train_df.columns.get_values()).copy()
        self.predictors.remove(self.target)
        self.predictors.remove('click_id')
        self.predictors.remove('minute')
        self.predictors.remove('second')
        print("categorical_features: {}\n".format(categorical_features))
        print("predictors: {}\n".format(self.predictors))
        # models.Optimization(model_name='lightgbm', train_df=self.train_df, validation_df=self.validation_df,
        #                    target=self.target, predictors=self.predictors, categorical=categorical_features)

        start_time = time.time()
        if model_name == 'lightgbm':
            self.get_model, self.get_model_estimator = models.lgbm_model(train_df=self.train_df,
                                                                         validation_df=self.validation_df,
                                                                         target=self.target,
                                                                         predictors=self.predictors,
                                                                         categorical=categorical_features)
        elif model_name == 'xgboost':
            self.get_model = models.xgboost_model(train_df=self.train_df,
                                                  validation_df=self.validation_df,
                                                  target=self.target,
                                                  predictors=self.predictors)
        elif model_name == 'catboost':
            categorical_index = [self.predictors.index(c) for c in categorical_features]
            self.get_model = models.catboost_model(train_df=self.train_df,
                                                   validation_df=self.validation_df,
                                                   target=self.target,
                                                   predictors=self.predictors,
                                                   categorical_index=categorical_index)
        else:
            pass
        end_time = time.time()
        print('Model training time: {}'.format(end_time - start_time))

    def ensemble_model(self, model_name):
        ensemble_data = pd.DataFrame()
        ensemble_data['target'] = self.validation_df[self.target]
        save_ensemble_path = str(self.local_data_path / 'ensemble_level1_{}.csv'.format(model_name))

        sub = pd.DataFrame()
        sub['click_id'] = self.test_df['click_id'].astype('int')
        save_path = str(self.local_data_path / (self.output_file + '_{}.csv'.format(model_name)))
        gc.collect()

        try:
            if model_name == 'lightgbm':
                ensemble_data['is_attributed_lightgbm'] = self.get_model.predict(self.validation_df[self.predictors])
                sub['is_attributed'] = self.get_model.predict(self.test_df[self.predictors])
            elif model_name == 'xgboost':
                densemble = xgb.DMatrix(self.validation_df[self.predictors])
                ensemble_data['is_attributed_xgboost'] = self.get_model.predict(densemble, ntree_limit=self.get_model.best_ntree_limit)
                dtest = xgb.DMatrix(self.test_df[self.predictors])
                sub['is_attributed'] = self.get_model.predict(dtest, ntree_limit=self.get_model.best_ntree_limit)
            elif model_name == 'catboost':
                ensemble_data['is_attributed_catboost'] = self.get_model.predict_proba(data=self.validation_df[self.predictors])[:, 1]
                sub['is_attributed'] = self.get_model.predict_proba(data=self.test_df[self.predictors])[:, 1]
            ensemble_data.to_csv(save_ensemble_path, index=False)
            sub.to_csv(save_path, index=False)
        except Exception as e:
            print(e)

        # output result
        output_ensemble_path = os.path.join(self.data_source_path, 'ensemble_data')
        cmd_sub = 'gsutil cp {} {}'.format(save_path, self.output_path)
        cmd_ensemble = 'gsutil cp {} {}'.format(save_ensemble_path, output_ensemble_path)
        for c in [cmd_ensemble, cmd_sub]:
            res = subprocess.check_call(c, shell=True)
            if not res:
                print('Output result file to GCS succeed...')
            else:
                print('Output result file failed, please check...')


    def prediction(self, model_name):
        sub = pd.DataFrame()
        sub['click_id'] = self.test_df['click_id'].astype('int')
        gc.collect()
        save_path = str(self.local_data_path / (self.output_file + '_{}.csv'.format(model_name)))
        try:
            print("Predicting...")
            if model_name == 'lightgbm':
                sub['is_attributed'] = self.get_model.predict(self.test_df[self.predictors])
            elif model_name == 'xgboost':
                dtest = xgb.DMatrix(self.test_df[self.predictors])
                sub['is_attributed'] = self.get_model.predict(dtest, ntree_limit=self.get_model.best_ntree_limit)
            elif model_name == 'catboost':
                sub['is_attributed'] = self.get_model.predict_proba(data=self.test_df[self.predictors])[:, 1]

            sub.to_csv(save_path, index=False)
            print("Done...")
        except Exception as e:
            print('[ERROR] {}, {}'.format(e, traceback.print_exc()))

        # output result
        cmd = 'gsutil cp {} {}'.format(save_path, self.output_path)
        res = subprocess.check_call(cmd, shell=True)
        if not res:
            print('Output prediction result file to GCS...')
        else:
            print('Output prediction result file failed, please check...')

    def run(self):
        if self.is_prepare_data:
            self.load_feature_data()
        else:
            self.copy_data_from_gcs()
            self.load_data()
            self.features_engineering(features_version=2)
        #self.split_data()
        #print('[INFO] Using {} model to train & predict'.format(self.mn))
        #self.train_model(model_name=self.mn)
        #self.ensemble_model(model_name=self.mn)
        #self.prediction(model_name=self.mn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle - talking data ad tracking fraud detection')
    parser.add_argument('--job-dir', action='store', dest='jd', type=str, required=True,
                        help='Project source path at GCS')
    parser.add_argument('--data-source', action='store', dest='ds', type=str, required=True,
                        help='Project source path at GCS')
    parser.add_argument('--output-file', action='store', dest='ofn', type=str, required=True,
                        help='Define output file name')
    parser.add_argument('--model-name', action='store', dest='mn', type=str, required=True,
                        help='Enter model name which want to use. ["lightgbm", "xgboost", "catboost"]')
    parser.add_argument('--is-debug', action='store', dest='isd', type=int, required=True,
                        help='Is debug mode or not')
    parser.add_argument('--is-prepare-data', action='store', dest='ispd', type=int, required=False, default=0,
                        help='Use feature data or not')
    args = parser.parse_args()
    np.random.seed(202109)  # for reproducibility
    total_nrows = 184903891 - 1
    if args.isd:
        set_train_rows = 1000000
        set_nchuck = 65000000
    else:
        set_train_rows = total_nrows
        set_nchuck = 65000000

    process = KaggleCompeitions(_data_source=args.ds, _output_file=args.ofn, _output_path=args.jd,
                                _train_rows=set_train_rows, _nchunk=set_nchuck, _total_rows=total_nrows,
                                _model_name=args.mn, _is_debug=args.isd, _is_prepare_data=args.ispd)
    process.run()
