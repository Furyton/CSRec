import logging
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.base import BaseModel

# def neg_sample(seq, labels, num_item, sample_size):
#     negs = set()
#     seen = set(labels)

#     while len(negs) < sample_size:
#         candidate = np.random.randint(0, num_item) + 1
#         while candidate in seen or candidate in negs:
#             candidate = np.random.randint(0, num_item) + 1
#         negs.add(candidate)
#     return negs
    # keys = range(1, num_item + 1)
    # sample_id = np.random.choice(keys, sample_size + len(seen), replace=False)
    # sample_ids = [x for x in sample_id if x not in seen]

    # return sample_ids[:sample_size]

class SoftLoss:
    r"""
        no sample
    """

    def __init__(self, mentor: BaseModel, args) -> None:
        self.mentor = mentor
        self.alpha = args.alpha
        self.T = args.T
        self.num_item = args.num_items
        self.device = args.device

        self.mentor.eval()

        self.nan = 0
        self.not_nan = 0
        self.debug = 0
        self.accum_iter = 0

    def debug_summary(self):
        logging.debug(f"loss nan summary: nan {self.nan} times, not nan {self.not_nan} times, ratio nan / (nan + not_nan) = {1.0 * self.nan / (self.nan + self.not_nan)}")

    def calculate_loss(self, model: BaseModel, batch):
        self.accum_iter += 1
        seqs = batch[0]
        labels = batch[1]

        with torch.no_grad():
            soft_target = self.mentor(batch)

        # output = (predict logits, predict loss, reg loss, ...)
        output = model.calculate_loss(batch)

        pred_logits = output[0]
        pred_loss = output[1]
        reg_loss = 0.

        if len(output) > 2:
            reg_loss = sum(output[2:])
        
        if len(pred_logits.size()) == 3:
            # batch_size * L * n_item, mask or auto regressive

            cl = labels[labels > 0]
            pred = pred_logits[labels > 0]

            assert(len(soft_target.size()) == 3)

            soft_target = soft_target[labels > 0]
        else:
            # batch_size * n_item, next item type

            cl = labels[labels > 0]
            pred = pred_logits

            if len(soft_target.size()) == 3:
                # B * L * N
                soft_target = soft_target[:, -1, :].squeeze()

        assert(soft_target.size() == pred_logits.size())

        cl_onehot = F.one_hot(cl, num_classes=self.num_item + 1)

        soft_target = 0.5 * ((soft_target / self.T).softmax(dim=-1) + cl_onehot)

        if self.accum_iter % 10000 < 5 and self.accum_iter != 0:
            if self.debug != 0:
                with torch.no_grad():
                    self.debug -= 1
                    
                    logging.debug(f"soft_target max in softmax: {soft_target.softmax(dim=-1).max()}, argmax {soft_target.softmax(dim=-1).argmax()}")
                    logging.debug(f"pred max in softmax: {pred.softmax(dim=-1).max()}, argmax {pred.softmax(dim=-1).argmax()}")

        KL_loss = F.kl_div(F.log_softmax(pred[:, 1:], dim=-1), soft_target[:, 1:], reduction='batchmean')

        if ~torch.isinf(KL_loss):
            loss = (1 - self.alpha) * pred_loss + self.alpha * KL_loss + reg_loss
            self.not_nan += 1
        else:
            loss = pred_loss + reg_loss
            self.nan += 1
        
        return loss

class BasicLoss:
    def __init__(self, **kwargs) -> None:
        pass

    def debug_summary(self):
        pass

    def calculate_loss(self, model: BaseModel, batch):
        # output = (predict logits, predict loss, reg loss, ...)
        output = model.calculate_loss(batch)

        return sum(output[1:])

class DVAELoss:
    r"""
    DVAELoss for multiclass

    note: L = \alpha KL(f||g) + (1-\alpha)KL(g||f) + h^Tf
    """
    def __init__(self, prior_model: BaseModel, auxiliary_model: BaseModel, args, trainable: bool=False) -> None:
        self.prior_model = prior_model
        self.auxiliary_model = auxiliary_model
        self.trainable = trainable

        self.alpha = args.dvae_alpha
        self.device = args.device

        if not self.trainable:
            self.auxiliary_model.eval()
        
        self.debug = 0

        self.nan = 0
        self.not_nan = 0

        self.accum_iter = 0

    def debug_summary(self):
        logging.debug(f"loss nan summary: nan {self.nan} times, not nan {self.not_nan} times, ratio nan / (nan + not_nan) = {1.0 * self.nan / (self.nan + self.not_nan)}")

    def calculate_loss(self, model: BaseModel, batch):
        self.accum_iter += 1
        seqs = batch[0]
        labels = batch[1]

        if not self.trainable:
            with torch.no_grad():
                g = self.auxiliary_model(batch)
        else:
            g = self.auxiliary_model(batch)

        h = self.prior_model(batch)

        output = model.calculate_loss(batch)

        f = output[0]
        pred_loss = output[1]
        reg_loss = 0.

        if len(output) > 2:
            reg_loss = sum(output[2:])

        assert(len(f.size()) == 2 and f.size() == h.size() and f.size() == g.size())

        f = f[:, 1:]
        g = g[:, 1:]
        h = h[:, 1:]

        # g: B x n_item
        # h: B x n_item
        # f: B x n_item

        KL_Loss_1 = F.kl_div(F.log_softmax(g, dim=-1), f, reduction='batchmean')
        KL_Loss_2 = F.kl_div(F.log_softmax(f, dim=-1), g, reduction='batchmean')

        KL_loss = self.alpha * KL_Loss_1 + (1. - self.alpha) * KL_Loss_2

        expectation_loss = (f.softmax(dim=-1) * h.softmax(dim=-1)).sum()

        if self.accum_iter % 10000 < 5 and self.accum_iter != 0:
            if self.debug != 0:
                with torch.no_grad():
                    self.debug -= 1

                    # logging.debug(f"f: \n" + str(f))
                    # logging.debug(f"g: \n" + str(g))
                    # logging.debug(f"h: \n" + str(h))

                    logging.debug(f"KL_loss1: \n" + str(KL_Loss_1))
                    logging.debug(f"KL_loss2: \n" + str(KL_Loss_2))
                    logging.debug(f"expectation_loss: \n" + str(expectation_loss))

                    logging.debug(f"max in g softmax: {g.softmax(dim=-1).max()} argmax: {g.softmax(dim=-1).argmax()}")
                    logging.debug(f"max in f softmax: {f.softmax(dim=-1).max()} argmax: {f.softmax(dim=-1).argmax()}")
                    logging.debug(f"max in h softmax: {h.softmax(dim=-1).max()} argmax: {h.softmax(dim=-1).argmax()}")

        if ~torch.isnan(KL_loss):
            self.not_nan += 1
            return pred_loss + expectation_loss + reg_loss
        else:
            self.nan += 1
            return pred_loss + KL_loss + expectation_loss + reg_loss


# class LE:
#     def __init__(self, model, args):
#         self.model = model
#         self.b = args.alpha

#         self.t = args.T
#         self.enable_sample = args.enable_sample
#         self.num_item = args.num_items
#         self.sample_ratio = args.samples_ratio
#         if self.enable_sample:
#             self.ce = nn.CrossEntropyLoss()
#         else:
#             self.ce = nn.CrossEntropyLoss(ignore_index=0)
#         self.device = args.device
#         self.debug = 0

#         # self.b = torch.tensor(0., device=self.device, requires_grad=True)

#     def compute(self, pred, batch):

#         # seqs, labels, rating = batch
#         seqs = batch[0]
#         labels = batch[1]

#         with torch.no_grad():
#             soft_target = self.model(batch).detach().clone()  # B * L * N or B * N

#         # we want both pred and soft_target to be 2-D tensor, like B * N, cl is a tensor which the size at 1st dim is the same, like B * 1

#         if len(pred.size()) == 3: 
#             # B * L * N, mask type
#             cl = labels[labels > 0]
#             pred = pred[labels > 0]

#             assert(len(soft_target.size()) == 3)

#             soft_target = soft_target[labels > 0]
#         else:
#             # B * N, next type
#             cl = labels.squeeze()
#             if len(soft_target.size()) == 3:
#                 # B * L * N
#                 soft_target = soft_target[:, -1, :].squeeze()
#             else:
#                 # B * N
#                 soft_target
        
#         assert(pred.size() == soft_target.size())

#         _KL_Loss = 0.
#         _CE_Loss = 0.

#         _A_KL_Loss = 0.
#         _A_CE_Loss = 0.

#         _al = 0.

#         if self.enable_sample:
#             logging.fatal("sampling in training stage has not been fully tested yet.")
#             raise NotImplementedError("not fully tested yet")
#             # 负样本采样 
#             negs = neg_sample(seqs, labels, self.num_item, int(self.num_item * self.sample_ratio))
#             negs = torch.LongTensor(list(negs)).repeat(len(cl), 1).to(self.device)
#             target = torch.cat((cl.unsqueeze(1), negs), 1)
#             # 采样后的one_hot
#             one_hot = [1] + [0] * negs.size(-1)
#             one_hot = torch.LongTensor(one_hot).repeat(negs.size(0), 1).to(torch.device(self.device))
#             # 抽取采样后的结果
#             pred = pred.gather(dim=1, index=target)
#             soft_target = soft_target.gather(dim=1, index=target)

#             # print(f"[loss self.enable_sample] soft_target.size(): {soft_target.size()}")

#             # 标签
#             label = torch.LongTensor([0] * pred.size(0)).to(torch.device(self.device))
#             soft_target = ((soft_target - soft_target.mean(dim=-1).unsqueeze(-1)) / soft_target.std(dim=-1).unsqueeze(-1))
#             # 计算kl的值
#             # KL_Loss = nn.functional.kl_div((pred.softmax(dim=-1) / self.t).log(), 0.5 * ((soft_target.softmax(dim=-1) / self.t) + one_hot), reduction='batchmean')
#             KL_Loss = nn.functional.kl_div(F.log_softmax(pred, dim=-1), 0.5 * ((soft_target / self.t).softmax(dim=-1) + one_hot),  reduction='batchmean')

#             if ~torch.isinf(KL_Loss):
#                 tmp_ce = self.ce(pred, label)
#                 # alpha = 0.2 * torch.sigmoid(self.b)
#                 alpha = self.b

#                 # _al = alpha.item()
#                 _al = alpha

#                 # tmp_kl = pow(self.t, 2) * alpha * KL_Loss
#                 tmp_kl = alpha * KL_Loss

#                 loss = (1 - alpha) * tmp_ce

#                 _KL_Loss = KL_Loss.item()
#                 _CE_Loss = tmp_ce.item()
#                 _A_CE_Loss = loss.item()
#                 _A_KL_Loss = tmp_kl.item()

#                 loss += tmp_kl
#             else:
#                 loss = self.ce(pred, label)
#                 _CE_Loss = _A_CE_Loss = loss.item()
#         else:
#             # 标签转换成 one_hot
#             cl_onehot = torch.nn.functional.one_hot(cl, num_classes=self.num_item + 1)

#             if self.debug == 0:
#                 self.debug = 1
#                 format_debug_str ="""
#                                 soft target: %s
#                                 max: %s
#                                 2nd value: %s
#                                 max - 2nd value: %s
#                                 after softmax: %s
#                                 max in softmax: %s
#                                 """

#                 logging.debug(format_debug_str, soft_target, soft_target.max(), soft_target.kthvalue(soft_target.size(-1) - 1).values.max(), (soft_target.max(dim=-1).values - soft_target.kthvalue(soft_target.size(-1) - 1).values).max().item(), soft_target.softmax(dim=-1), soft_target.softmax(dim=-1).max())

#             soft_target: torch.Tensor
#             # soft target logit value varies too much, need to normalize a bit

#             KL_Loss = nn.functional.kl_div(F.log_softmax(pred, dim=-1), 0.5 * ((soft_target / self.t).softmax(dim=-1) + cl_onehot), reduction='batchmean')

#             if ~torch.isinf(KL_Loss):
#                 tmp_ce = self.ce(pred, cl)
#                 alpha = self.b
#                 _al = alpha

#                 tmp_kl = alpha * KL_Loss

#                 loss = (1 - alpha) * tmp_ce

#                 _KL_Loss = KL_Loss.item()
#                 _CE_Loss = tmp_ce.item()
#                 _A_CE_Loss = loss.item()
#                 _A_KL_Loss = tmp_kl.item()

#                 loss += tmp_kl
#             else:
#                 loss = self.ce(pred, cl)
#                 _CE_Loss = _A_CE_Loss = loss.item()

#         return loss, _CE_Loss, _KL_Loss, _A_CE_Loss, _A_KL_Loss, _al


# class CELoss:
#     def __init__(self, args):
#         self.enable_sample = args.enable_sample
#         self.num_item = args.num_items
#         self.sample_ratio = args.samples_ratio
#         if self.enable_sample:
#             self.ce = nn.CrossEntropyLoss()
#         else:
#             self.ce = nn.CrossEntropyLoss(ignore_index=0)
#         self.device = args.device

#     def compute(self, pred, batch):
#         # seqs, labels, rating = batch
#         seqs = batch[0]
#         labels = batch[1]

#         if isinstance(pred, tuple):
#             pred = sum(pred)

#         if len(pred.size()) == 3:
#             cl = labels[labels > 0]
#             pred = pred[labels > 0]
#         else:
#             cl = labels.squeeze()

#         if self.enable_sample:
#             logging.fatal("sampling in training stage has not been fully tested yet.")
#             raise NotImplementedError("not fully tested yet")
#             # 负样本采样 
#             negs = neg_sample(seqs, labels, self.num_item, int(self.num_item * self.sample_ratio))
#             negs = torch.LongTensor(list(negs)).repeat(len(cl), 1).to(self.device)
#             target = torch.cat((cl.unsqueeze(1), negs), 1)
#             # 采样后的one_hot
#             one_hot = [1] + [0] * negs.size(-1)
#             one_hot = torch.LongTensor(one_hot).repeat(negs.size(0), 1).to(torch.device(self.device))
#             # 抽取采样后的结果
#             pred = pred.gather(dim=1, index=target)

#             # 标签
#             label = torch.LongTensor([0] * pred.size(0)).to(torch.device(self.device))
            
#             loss = self.ce(pred, label)

#         else:
#             # 标签转换成 one_hot
#             # cl_onehot = torch.nn.functional.one_hot(cl, num_classes=self.num_item + 1)
#             loss = self.ce(pred, cl)

#         return loss
