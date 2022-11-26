import os
import pickle
import logging
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from configuration.config import *
from sklearn.utils import shuffle

from utils import get_path, set_temporary_np_seed_as


def _remap(df: pd.DataFrame, columns: list):
    # print("[INFO: reindex]")
    # logging.debug("do_remap")
    with set_temporary_np_seed_as(2022):
        for column in columns:
            df.loc[:, column] = df[column].map(
                dict(zip(shuffle(df[column].unique()), range(1, len(df[column].unique()) + 1))))


def _drop_cold(df: pd.DataFrame, min_user: int, min_item: int, do_remap: bool = True):
    logging.info(
        f"filtering out cold users (who interacts with less or eq than {min_user} items) and cold items (which is interacted by less or eq than {min_item} users)")

    logging.debug(
        f"before filtering, there are {len(df[SESSION_ID].unique())} users, {len(df[ITEM_ID].unique())} items ")

    max_iter = 20

    while True:
        max_iter -= 1
        if max_iter <= 0:
            logging.fatal("iterated too many times (20 times). please consider another denser dataset.")
            raise RecursionError("iterated too many times (10 times). please consider another denser dataset.")

        user_cnt = df.groupby(SESSION_ID).count()
        cold_user_id = user_cnt[user_cnt[RATING] < min_user].index

        item_cnt = df.groupby(ITEM_ID).count()
        cold_item_id = item_cnt[item_cnt[RATING] < min_item].index

        if len(cold_user_id) == 0 and len(cold_item_id) == 0:
            logging.debug(
                f"after {20 - max_iter - 1} filterings, there are {len(df[SESSION_ID].unique())} users, {len(df[ITEM_ID].unique())} items")

            # logging.info(f"user desc")

            logging.info(
                "user desc \n {}".format(user_cnt.drop(columns=ITEM_ID).rename(columns={RATING: "seq_len"}).describe()))

            # logging.info(f"item desc")

            logging.info("item desc \n {}".format(
                item_cnt.drop(columns=SESSION_ID).rename(columns={RATING: "item_pop"}).describe()))

            return df.copy()

        if len(cold_user_id) > 0:
            df = df.drop(index=df[df[SESSION_ID].isin(cold_user_id)].index).reset_index(drop=True).copy()

        if len(cold_item_id) > 0:
            df = df.drop(index=df[df[ITEM_ID].isin(cold_item_id)].index).reset_index(drop=True).copy()

        if do_remap:
            _remap(df, [SESSION_ID, ITEM_ID])
        else:
            _remap(df, [SESSION_ID])


def _df_data_partition(dataframe: pd.DataFrame, max_len: int, prop_sliding_window: float, use_rating=True, min_length=5, min_item_inter=5, do_remap=True, pd_itemnum=None, only_good=False) -> list:

    if not dataframe.columns.isin([SESSION_ID, ITEM_ID, RATING, TIMESTAMP]).all():
        logging.fatal(
            f"illegal dataset format, expect colomns to be {[SESSION_ID, ITEM_ID, RATING, TIMESTAMP]}, but got {dataframe.columns}")
        raise ValueError

    if only_good:
        logging.info("filtering out items user dislike.")
        dataframe = dataframe[dataframe[RATING] == 1].copy()

    if do_remap:
        _remap(dataframe, [SESSION_ID, ITEM_ID])
    else:
        _remap(dataframe, [SESSION_ID])

    if dataframe.columns.isin([TIMESTAMP]).any():
        logging.info("sorting according to timestamp")
        if dataframe[TIMESTAMP].dtype == object:
            dataframe[TIMESTAMP] = dataframe[TIMESTAMP].apply(lambda x: pd.to_datetime(x).timestamp()).astype(int)

        dataframe = dataframe.sort_values(by=[SESSION_ID, TIMESTAMP], ignore_index=True).drop(columns=TIMESTAMP).copy()

    dataframe = _drop_cold(dataframe, min_length, min_item_inter, do_remap)

    # logging.info('dataset summary:')
    logging.info(f'dataset summary:\n{dataframe.describe()}')

    if prop_sliding_window >= 1.0:
        sliding_step = int(prop_sliding_window)
    elif prop_sliding_window > 0:
        sliding_step = int(prop_sliding_window * max_len)
    elif prop_sliding_window == -1.0:
        sliding_step = max_len
    else:
        logging.critical(f"illegal prop_sliding_window value{prop_sliding_window}")
        raise ValueError

    # sliding_step = int(prop_sliding_window * max_len) if prop_sliding_window != -1.0 else max_len
    # sliding_step = 20 

    def process(df: pd.DataFrame, column, data):
        s = df.loc[:, column]
        if len(s) <= max_len:
            data.append(list(s.to_numpy()))
        else:
            beg_idx = range(len(s) - max_len, 0, -sliding_step)

            if beg_idx[0] != 0:
                data.append(list(s.iloc[0:max_len].to_numpy()))

            for i in beg_idx[::-1]:
                data.append(list(s.iloc[i:i + max_len].to_numpy()))

    sessoin_group = dataframe.groupby(SESSION_ID)

    item_data = []

    sessoin_group.apply(lambda x: process(x, ITEM_ID, item_data))

    itemnum = _get_itemnum(dataframe) if pd_itemnum is None else pd_itemnum

    usernum = len(item_data)

    item_train, item_valid, item_test = [], [], []

    for seq in item_data:
        item_train.append(seq[:-2])
        item_valid.append([seq[-2]])
        item_test.append([seq[-1]])

    if use_rating is True:
        rating_data = []
        rating_train, rating_valid, rating_test = [], [], []
        sessoin_group.apply(lambda x: process(x, RATING, rating_data))

        for seq in rating_data:
            rating_train.append(seq[:-2])
            rating_valid.append([seq[-2]])
            rating_test.append([seq[-1]])

        return [item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test]
    else:
        return [item_train, item_valid, item_test, usernum, itemnum]


def _get_itemnum(dataframe: pd.DataFrame) -> int:
    if not dataframe.columns.isin([ITEM_ID]).any():
        logging.fatal(f"illegal dataset format, expect colomns to be {[ITEM_ID]}, but got {dataframe.columns}")
        raise ValueError
    itemnum = len(dataframe[ITEM_ID].unique())

    return itemnum

def _sample_from_dataset(dataset: list, rate: float, use_rating: bool=False, seed: int=0):
    if not use_rating:
        item_train, item_valid, item_test, usernum, itemnum = dataset[:5]
    else:
        item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test = dataset
    
    n_sampled_user = int(usernum * rate)
    with set_temporary_np_seed_as(seed):
        selected_users = list(np.random.choice(range(usernum), n_sampled_user, replace=False))
    
    def _select_from(a: list, indices: list):
        tmp_array = np.array(a, dtype=list)
        selected_list = np.ndarray.tolist(tmp_array[indices])
        del tmp_array
        return selected_list
    
    new_item_train = _select_from(item_train, selected_users)
    new_item_valid = _select_from(item_valid, selected_users)
    new_item_test = _select_from(item_test, selected_users)

    logging.info(f"#users after sampling: {len(new_item_train)}, #item after sampling: {itemnum}")

    if not use_rating:
        return [new_item_train, new_item_valid, new_item_test, n_sampled_user, itemnum]
    else:
        new_rating_train = _select_from(rating_train, selected_users)
        new_rating_valid = _select_from(rating_valid, selected_users)
        new_rating_test = _select_from(rating_test, selected_users)

        return [new_item_train, new_item_valid, new_item_test, n_sampled_user, itemnum, new_rating_train, new_rating_valid, new_rating_test]

# header contains dataset_name, min_user, min_item, good only?, do_remap?, use_rating?
def _check_dataset_cache(args, header, fully_check=True) -> bool:
    logging.info('check if the cache is generated under this configuration')

    def _warning_report(field_name, field1, field2):
        logging.warning(
            f'{field_name}: {field1} and {field2} maybe different configurations? I refuse to use this cache.')

    basic_keys = {'dataset_name', 'min_user', 'min_item', 'good_only', 'do_reindex', 'use_rating'}
    sampling_keys = {'sample_rate', 'sample_seed'}
    current_header = _gen_cache_header(args)

    for entry in basic_keys:
        try:
            if current_header[entry] != header[entry]:
                _warning_report(entry, current_header[entry], header[entry])
                return False
        except:
            logging.warning(f'failed when checking field {entry} when checking headers')
            return False

    if not fully_check:
        logging.info('correct.')
        return True
    
    if args.do_sampling and ('do_sampling' not in header or not header['do_sampling']):
        _warning_report('do_sampling', True, False)
        return False

    if not args.do_sampling and ('do_sampling' in header and header['do_sampling']):
        _warning_report('do_sampling', False, True)
        return False

    if args.do_sampling:
        for entry in sampling_keys:
            try:
                if current_header[entry] != header[entry]:
                    _warning_report(entry, current_header[entry], header[entry])
                    return False
            except:
                logging.warning(f'failed when checking field {entry} when checking headers')
                return False

    logging.info('correct.')
    return True


def _gen_dataset(args) -> list:
    logging.info(f'processing dataset {args.dataset_name}')

    current_directory = path.dirname(__file__)
    parent_directory = path.split(current_directory)[0]
    dataset_filepath = path.join(parent_directory, RAW_DATASET_ROOT_FOLDER, args.dataset_name)
    data = pd.read_csv(dataset_filepath)
    # dataset = df_data_partition(args, data, use_rating=args.use_rating, do_remap=args.do_remap,
                                # pd_itemnum=args.num_items, only_good=args.good_only)
    dataset = _df_data_partition(data, max_len=args.max_len, prop_sliding_window=args.prop_sliding_window, use_rating=args.use_rating, min_length=args.min_length, min_item_inter=args.min_item_inter, do_remap=args.do_remap, pd_itemnum=args.num_items, only_good=args.good_only)

    args.num_items = dataset[4]

    return dataset


def _gen_cache_path(args, record_sample_info=True) -> Path:
    current_directory = path.dirname(__file__)
    parent_directory = path.split(current_directory)[0]

    if args.do_sampling and record_sample_info:
        cache_filename = args.dataset_cache_filename or 'sampled_{}-{}-{}-rate-{}-seed-{}.pkl'.format(args.dataset_name.split('.')[0], args.min_length, args.min_item_inter, args.sample_rate, args.sample_seed)
    else:
        cache_filename = args.dataset_cache_filename or '{}-{}-{}.pkl'.format(args.dataset_name.split('.')[0],
                                                                          args.min_length, args.min_item_inter)

    folder = Path(parent_directory).joinpath(RAW_DATASET_ROOT_FOLDER, PROCESSED_DATASET_CACHE_FOLDER)

    os.makedirs(folder, exist_ok=True)

    filename = folder.joinpath(cache_filename)

    return filename

# cache processed dataset

def _gen_cache_header(args, record_sample_info=True):
    header = {'dataset_name': args.dataset_name,
            'min_user': args.min_length,
            'min_item': args.min_item_inter,
            'good_only': args.good_only,
            'do_reindex': args.do_remap,
            'use_rating': args.use_rating}
    
    if not record_sample_info:
        return header
    
    header.update({'do_sampling': args.do_sampling})

    if args.do_sampling:
        header.update({'sample_seed': args.sample_seed,
                       'sample_rate': args.sample_rate})
    
    return header

def _cache_dataset(args, dataset, record_sample_info=True):
    header = _gen_cache_header(args, record_sample_info)

    cache_path = _gen_cache_path(args, record_sample_info)

    with cache_path.open('wb') as f:
        pickle.dump((header, dataset), f)

def _load_full_dataset_from_path(args, cache_path: Path, allow_regenerate=False):
    if not cache_path.exists():
        if not allow_regenerate:
            logging.critical(f"file from {cache_path} not found")
            raise FileNotFoundError

        logging.warning(f"file from {cache_path} not found. Regenerating.")

        dataset = _gen_dataset(args)
        _cache_dataset(args, dataset, record_sample_info=False)
    else:
        if cache_path.is_file():
            logging.info(f"loading cache from {cache_path}")
            dataset_cache = pickle.load(cache_path.open('rb'))

            header, dataset = dataset_cache

            if not _check_dataset_cache(args, header, fully_check=False):
                if not allow_regenerate:
                    logging.critical('bad cache detected')
                    raise ValueError

                logging.warning('bad cache detected. regenerating')

                dataset = _gen_dataset(args)

                _cache_dataset(args, dataset, record_sample_info=False)
        else:
            logging.fatal(f"{cache_path} is not a file.")
            raise ValueError("cache path is not a file")
        
    return dataset

def _resampling(args):
    if args.path_for_sample is None:
        logging.warning(f"sample data from scratch")
        full_dataset = _gen_dataset(args)
    else:
        logging.debug(f'loading full dataset from {args.path_for_sample}')

        path_ready_for_sampling = get_path(args.path_for_sample)

        full_dataset = _load_full_dataset_from_path(args, path_ready_for_sampling)

    sampled_dataset = _sample_from_dataset(full_dataset, args.sample_rate, args.use_rating, args.sample_seed)

    _cache_dataset(args, sampled_dataset)

    return sampled_dataset

def _load_sampled_dataset_from_path(args, cache_path: Path, allow_regenerate=True):
    if not cache_path.exists():
        if not allow_regenerate:
            logging.critical(f"file from {cache_path} not found")
            raise FileNotFoundError
        
        logging.warning(f"file from {cache_path} not found. Regenerating.")

        sampled_dataset = _resampling(args)
    else:
        if cache_path.is_file():
            logging.info(f"loading sampled dataset from {cache_path}")

            dataset_cache = pickle.load(cache_path.open('rb'))

            header, sampled_dataset = dataset_cache

            if not _check_dataset_cache(args, header):
                if not allow_regenerate:
                    logging.critical('bad cache detected')
                    raise ValueError
                logging.warning('bad cache detected. regenerating')
                sampled_dataset = _resampling(args)
        else:
            logging.fatal(f"{cache_path} is not a file.")
            raise ValueError("cache path is not a file")
    
    return sampled_dataset

def get_dataset(args):
    if not args.do_sampling:
        if args.load_processed_dataset:
            cache_file_path = _gen_cache_path(args)
            dataset = _load_full_dataset_from_path(args, cache_file_path)
        elif args.save_processed_dataset:
            dataset = _gen_dataset(args)

            _cache_dataset(args, dataset)
        else:
            dataset = _gen_dataset(args)
    else:
        if args.load_processed_dataset:
            cache_file_path = _gen_cache_path(args)
    
            dataset = _load_sampled_dataset_from_path(args, cache_file_path)
        elif args.save_processed_dataset:
            dataset = _resampling(args)

            _cache_dataset(args, dataset)
        else:
            dataset = _resampling(args)
    
    return dataset
