import math
import torch

from collections import Counter

from models.base import BaseModel

class POPModel(BaseModel):
    def __init__(self, args, dataset, device, max_len):
        super(POPModel, self).__init__(dataset, device, max_len)

        # module with no parameter will cause a lot of problems
        # it is easy to solve, but takes a lot special handlings
        # adding a fake param seems to be an elegent solution :)

        self.fake_parameter = torch.nn.Linear(1, 1, False)

        item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test = dataset

        self.usernum = usernum
        self.itemnum = itemnum
        self.item_train = item_train
        self.item_valid = item_valid
        self.item_test = item_test

        self.popularity = self.items_by_popularity()

        pop = [0] * (self.itemnum + 1)

        for i, v in self.popularity.items():
            # pop[i] = math.log2( v + 2. )
            pop[i] = v

        # item id starts at 1

        self.pop = torch.tensor(pop, dtype=torch.float, device=self._device, requires_grad=False)

        # [a, b] -> [-1, 1]
        # ( x - (a + b) / 2 ) / ((a + b) / 2) = 2 * x / (a + b) - 1

        # [-5, 5]

        self.pop_distr = 5 * (2 * self.pop / (self.pop.max() + self.pop.min()) - 1)
    @classmethod
    def code(cls):
        return 'pop'

    def forward(self, batch):
        """
        input:
            batch: B x T
        output:
            B x N
        """

        x = batch[0]
        batch_size = x.size()[0]
        # length = x.size()[1]

        logits = self.pop_distr.repeat(batch_size, 1)

        return logits + 0 * self.fake_parameter.weight

    def items_by_popularity(self):
        popularity = Counter() 

        for user in range(0, self.usernum):
            popularity.update(self.item_train[user])
            popularity.update(self.item_valid[user])
            popularity.update(self.item_test[user])
        
        return popularity
    
    @torch.no_grad()
    def predict(self, batch):
        # seqs, candidates, labels, seq_lens, user = batch
        candidates = batch[1]
        scores = self.forward(batch)  # B x V
        # scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        return scores

    @torch.no_grad()
    def full_sort_predict(self, batch):
        scores= self.forward(batch)  # B x V
        # scores = scores[:, -1, :]  # B x V
        return scores
    
    def calculate_loss(self, batch):
        logits = self.forward(batch)

        return logits, self.fake_parameter.weight