from dataloaders.utils import get_dataset

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
        required args:
            - dataloader_factory():
                - num_items
                - dataloader_type
            - get_dataset():
                - do_sampling
                - dataset_cache_filename
                - load_processed_dataset
                - save_processed_dataset
                - min_item_inter
                - path_for_sample
                - max_len
                - min_length
                - do_remap
                - good_only
                - do_reindex
                - use_rating
                - sample_rate
                - sample_seed
    return:
        train, val, test, dataset

        train, val, test are DataLoaders

        dataset is a list,
            [item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test] or
            
            [item_train, item_valid, item_test, usernum, itemnum]
    """

    dataset = get_dataset(args)

    args.num_items = dataset[4]

    dataloader_ = DATALOADERS[args.dataloader_type]

    dataloader = dataloader_(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()

    return train, val, test, dataset
