import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from net import Net
from src.utils import save_model
import wandb


def calc_covariance_matrix(z1, z2):
    return (z1.T @ z2) / \
           (torch.sqrt((z1 ** 2).sum(0)) * torch.sqrt((z2 ** 2).sum(0)).reshape(-1, 1))


class BTWINS(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.net = Net(model_name)

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

    def __loss(self, batch):
        t1, t2 = self.transforms(batch), self.transforms(batch)
        z1, z2 = self.net(t1)[1], self.net(t2)[1]

        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        covariance_matrix = calc_covariance_matrix(z1, z2)

        invariance_loss = ((1 - covariance_matrix.diag()) ** 2).sum()
        off_diag_elements = self.get_off_diag_elements(covariance_matrix)

        return invariance_loss + 0.01 * (off_diag_elements ** 2).sum()

    def fit(self, trainloader, epochs=10, lr=1e-4):
        optim = Adam(self.parameters(), lr=lr)

        step = 0
        for epoch in range(epochs):
            pbar = tqdm(trainloader, desc='Training...')
            for batch_num, (batch, _) in enumerate(pbar):
                step += 1
                pbar.set_postfix({'epoch': epoch})
                batch = batch.to(self.device)
                loss = self.__loss(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                wandb.log({'loss': loss.item(), 'epoch': epoch + batch_num / len(trainloader)}, step=step)
            if not (epoch+1) % 3:
                save_model(self.net.model, epoch)
