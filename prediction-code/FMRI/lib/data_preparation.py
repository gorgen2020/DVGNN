# -*- coding:utf-8 -*-

import numpy as np

from .utils import get_sample_indices


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0,2,1,3)
    val = (val).transpose(0,2,1,3)
    test =(test).transpose(0,2,1,3)

    return {'mean': mean, 'std': std}, train, val, test


def read_and_generate_dataset(graph_signal_matrix_filename,generated_adj_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data
    merge: boolean, default False,
           whether to merge training set and validation set to train model
    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']

    #FMRI
    data_seq = data_seq[1:data_seq.shape[0],:,:]

    adj = np.load(generated_adj_filename)
    adj = np.array(adj,dtype=np.float16)
    len1 = data_seq.shape[0]

    for ii in range(len1):
        adj[ii]= adj[ii] + np.eye(data_seq.shape[1])




    all_samples = []
    all_adj = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq,adj, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target, hour_sample_adj = sample
        aa =  hour_sample_adj
        hour_sample_adj = hour_sample_adj[hour_sample_adj.shape[0]-1,:,:]
        all_adj.append((
            np.expand_dims(hour_sample_adj, axis=0),
            np.expand_dims(hour_sample_adj, axis=0),
            np.expand_dims(hour_sample_adj, axis=0),
            np.expand_dims(hour_sample_adj, axis=0)
        ))
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))

    fd = len(all_samples)
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
        training_set_adj = [np.concatenate(i, axis=0) for i in zip(*all_adj[:split_line1])]

    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]
        training_set_adj = [np.concatenate(i, axis=0)
                        for i in zip(*all_adj[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    validation_set_adj = [np.concatenate(i, axis=0)
                      for i in zip(*all_adj[split_line1: split_line2])]

    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    testing_set_adj = [np.concatenate(i, axis=0)
                   for i in zip(*all_adj[split_line2:])]


    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    train_hour_adj = training_set_adj[0]
    val_hour_adj = validation_set_adj[0]
    test_hour_adj = testing_set_adj[0]

    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    print('training data_adj:  recent: {}'.format(train_hour_adj.shape))
    print('validation data_adj:  recent: {}'.format(val_hour_adj.shape))
    print('testing data_adj: recent: {}'.format(test_hour_adj.shape))



    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)





    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
            'recent_adj': train_hour_adj,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target,
            'recent_adj': val_hour_adj,
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target,
            'recent_adj': test_hour_adj,
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data
