from dataloaders.utils import *

from dataloaders.MaskDataloader import MaskDataloader
from dataloaders.NextItemDataloader import NextItemDataloader


DATALOADERS = {
    MaskDataloader.code(): MaskDataloader,
    NextItemDataloader.code(): NextItemDataloader,
}

def dataloader_factory(args):
    """
    input:
        args: config
    return:
        train, val, test, dataset

        train, val, test are DataLoaders

        dataset is a list,
            [item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test] or
            
            [item_train, item_valid, item_test, usernum, itemnum]
    """

    if args.load_processed_dataset:
        cache_file_path = gen_cache_path(args)

        if not cache_file_path.exists():
            logging.warning('cache file not found. regenerating')

            dataset = gen_dataset(args)

            cache_dataset(args, dataset)
        else:
            if cache_file_path.is_file():
                logging.info(f"loading processed dataset cache in {cache_file_path}")
                dataset_cache = pickle.load(cache_file_path.open('rb'))

                header, dataset = dataset_cache

                args.num_items = dataset[4]

                if not check_dataset_cache(args, header):
                    logging.warning('bad cache detected. regenerating')

                    dataset = gen_dataset(args)

                    cache_dataset(args, dataset)
            else:
                logging.fatal(f"{cache_file_path} is not a file.")
                raise ValueError("not a file")
    elif args.save_processed_dataset:
        dataset = gen_dataset(args)

        cache_dataset(args, dataset)
    else:
        dataset = gen_dataset(args)

    dataloader_ = DATALOADERS[args.dataloader_type]

    dataloader = dataloader_(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()

    return train, val, test, dataset
