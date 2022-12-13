import torch
from tqdm import tqdm


class Train(object):

    def __init__(self, dataloader, model, loss, optimizer, metrics=None):
        self.dl = dataloader
        self.model = model
        self.metrics = metrics
        self.optimizer = optimizer
        self.loss = loss

    # def one_step(self, data):

    def fit(self, data=None, *args, **kwargs):

        data = data if data else self.dl
        for origin_tokens, masked_tokens, target_tokens in tqdm(self.dl):
            result = self.model(masked_tokens)['logits']
            loss = self.loss(
                result.view(-1, result.size(-1)),
                target_tokens.view(-1),
                reduction=kwargs["reduction"],
                ignore_index=kwargs["ingore_index"]
            )
            loss.backward()
            self.optimizer.step()
            print(loss)

    def eval(self, *args, **kwargs):
        self.model.eval()
        with torch.no_grad():
            for origin_tokens, masked_tokens, target_tokens in tqdm(self.dataloader):
                result = self.model(masked_tokens)['logits']
                loss = self.loss(
                    result.view(-1, result.size(-1)),
                    target_tokens.view(-1),
                    reduction=kwargs["reduction"],
                    ignore_index=kwargs["ingore_index"]
                )
                print(loss)
