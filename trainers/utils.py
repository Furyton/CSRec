import logging
import torch

def get_model_size(model):
    """
    args:
        model: torch.nn.Module
    return:
        param size: float (MB)
    """

    para_num = sum([p.numel() for p in model.parameters()])
	# para_size:  MB
    para_size = para_num * 4 / 1024 / 1024
    return para_size

def assert_model_device(model: torch.nn.Module, device: str, tag: str, device_idx):
    param_size = get_model_size(model)
    logging.info(f"model {tag} has {param_size} MB params.")

    if param_size == 0:
        logging.info(f"model {tag} has no param")

        model.to(device)
    else:
        def _get_model_device():
            return next(model.parameters()).device
        
        model_device = _get_model_device()

        if model_device != torch.device(f"{device}:{device_idx}"):
            logging.warning(f"model {tag} is not on the expected device: got: {model_device}, expected: {device}:{device_idx}")

            model.to(torch.device(device))


def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def _cal_metrics(labels, scores, metrics, ks, star=''):
    answer_count = labels.sum(1).to(torch.int)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall' + star + '@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float()).to(scores.device)
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(scores.device)
        ndcg = (dcg / idcg).mean().item()
        metrics['NDCG' + star + '@%d' % k] = ndcg

        position_mrr = torch.arange(1, k + 1).to(scores.device)
        weights_mrr = 1 / position_mrr.float()
        mrr = (hits * weights_mrr).sum(1)
        mrr = mrr.mean().item()

        metrics['MRR' + star + '@%d' % k] = mrr


# B x C, B x C
def recalls_ndcgs_and_mrr_for_ks(scores, labels, ks, ratings):
    metrics = {}

    _cal_metrics(labels, scores, metrics, ks)

    ratings = ratings.squeeze()
    scores = scores[ratings == 1.]
    labels = labels[ratings == 1.]

    _cal_metrics(labels, scores, metrics, ks, '*')

    return metrics
