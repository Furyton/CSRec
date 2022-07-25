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
