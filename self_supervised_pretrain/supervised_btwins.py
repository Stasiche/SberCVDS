import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from net import Net
from src.utils import save_model
import wandb
from src.utils import eval_model


def calc_covariance_matrix(z1, z2):
    return (z1.T @ z2) / \
           (torch.sqrt((z1 ** 2).sum(0)) * torch.sqrt((z2 ** 2).sum(0)).reshape(-1, 1))


class BTWINS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = Net(config.model_name)
        self.criterion = nn.CrossEntropyLoss()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(9),
                transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),
                transforms.RandomGrayscale(0.2),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def get_off_diag_elements(self, matrix):
        size = self.net.model.classifier[1].in_features
        mask = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
        return matrix.masked_select(mask)

    @property
    def device(self):
        return next(self.parameters()).device

    def __ss_loss(self, batch):
        t1, t2 = self.transforms(batch), self.transforms(batch)
        z1, z2 = self.net(t1)[1], self.net(t2)[1]

        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        covariance_matrix = calc_covariance_matrix(z1, z2)

        invariance_loss = ((1 - covariance_matrix.diag()) ** 2).sum()
        off_diag_elements = self.get_off_diag_elements(covariance_matrix)

        return invariance_loss + 0.01 * (off_diag_elements ** 2).sum()

    def __s_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def train_one_epoch(self, optimizer, epoch, step, trainloader):
        total_loss = 0
        for batch_num, (batch, labels) in enumerate(tqdm(trainloader, desc='Training...')):
            step += 1
            batch = batch.to(self.device)
            ssloss = self.__loss(batch)
            sloss = self.__s_loss(self.net(batch)[0], labels)

            loss = (1 - self.config.alpha) * ssloss + self.config.alpha * sloss
            loss /= self.config.grad_accum
            loss.backward()

            total_loss += loss.item()
            if not step % self.config.grad_accum:
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({'loss': total_loss}, step=step)
        wandb.log({'epoch': epoch + batch_num / len(trainloader)}, step=step)

        return step

    def fit(self, trainloader, valloader):
        optim = Adam(self.parameters(), lr=self.config.lr)

        step = 0
        # eval_model(model, valdata, step)
        for epoch in range(self.config.epochs):
            step = self.train_one_epoch(optim, epoch, step, trainloader)
            eval_model(self.model, valloader, step)
            if not (epoch + 1) % 2:
                save_model(self.model, epoch)
