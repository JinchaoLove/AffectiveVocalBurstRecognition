import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import CCC
# from utils.metrics import CCC_Aria as CCC
from models.utils import get_similarity, get_contrast_loss, get_softmax_loss

def compute_kl_loss(p, q, pad_mask=None):
    """ Note: KL-loss is only for discrete distribution. """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

def r_drop_loss(model, loss_cls, inputs, label, alpha=0.5):
    logits = model(inputs)
    if isinstance(logits, tuple):
        logits, logits2 = logits  # could use part of model only...
    else:
        logits2 = model(inputs)
    base_loss = 0.5 * (loss_cls(logits, label) + loss_cls(logits2, label))
    kl_loss = compute_kl_loss(logits, logits2)
    # carefully choose hyper-parameters
    loss = base_loss + alpha * kl_loss
    return loss


class CCC_Rdrop_Loss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.ccc = CCCLoss(torch.Tensor(cfg.loss_weight))
        self.diff = nn.MSELoss()
        self.alpha = cfg.loss_alpha if (cfg is not None and 'loss_alpha' in cfg.keys() and cfg.loss_alpha) else 0.5

    def forward(self, pred, batch):
        # y_pred: [#B, #n_emo] after sigmoid
        if not isinstance(pred, (list, tuple)):
            return{'loss': CCC(pred, batch['labels'])}
        assert len(pred) == 2, f"should have 2 preds but get {len(pred)}."
        pred1, pred2 = pred
        ccc1 = self.ccc(pred1, batch['labels'])
        ccc2 = self.ccc(pred2, batch['labels'])
        ccc_loss = (ccc1 + ccc2) * 0.5
        # r-drop
        diff_loss = self.diff(pred1, pred2)
        loss = ccc_loss + self.alpha * diff_loss
        return {'loss': loss, 'ccc_loss': ccc_loss, 'diff_loss': diff_loss}


class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ccc = CCCLoss()

    def forward(self, pred, labels):
        return{
            'loss': self.ccc(pred, labels),
        }


class DALoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = CCCLoss()
        self.ge2e = GE2ELoss()

    def forward(self, pred, gt, spkr_emb, spkr):
        l1_loss = self.l1(pred, gt)
        spkr_loss = self.ge2e(spkr_emb)
        loss = l1_loss + spkr_loss
        return dict(loss=loss, l1_loss=l1_loss, spkr_loss=spkr_loss)


class GE2ELoss(nn.Module):
    """ Speaker verification. """
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)
        self.loss_type = 'softmax'

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        similarity = get_similarity(embeddings)
        #print(similarity[0, 0, :])
        similarity = self.w * similarity + self.b
        if self.loss_type == 'contrast':
            loss = get_contrast_loss(similarity)
        else:
            loss = get_softmax_loss(similarity)

        return loss

class CCCLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        if weight is not None:
            self.weight = nn.Sequential(nn.Parameter(weight), nn.Softmax(dim=-1))
        else:
            self.weight = None

    def forward(self, pred, gt):
        assert pred.shape == gt.shape
        if pred.dim() > 2:  # gather: (bsz, n_labels) -> (world_size, bsz, n_labels)
            pred = pred.reshape(-1, pred.shape[-1])
            gt = gt.reshape(-1, gt.shape[-1])
        ccc, single = CCC(pred, gt, True)
        if self.weight is not None:
            ccc = (single * self.weight.type_as(ccc)).sum()
        loss = 1 - ccc
        return loss

class ShrinkageLoss(nn.Module):
    '''
        Shrinkage Loss for regression task
        Args:
            a: shrinkage speed factor
            c: shrinkage position
    '''

    def __init__(self, a, c):
        super().__init__()
        self.a = a
        self.c = c

    def forward(self, pred, gt):
        l1 = torch.abs(pred - gt)
        loss = l1**2 / (1 + torch.exp(self.a * (self.c - l1)))
        loss = loss.mean()
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, gt):
        '''
            Args:
                pred: [#B, #C]
                gt: [#B, #C]
        '''
        assert pred.dim() == 2 and pred.shape == gt.shape, f'Pred shape {pred.shape} is not equal to gt shape {gt.shape}'
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        gt_diff = gt.unsqueeze(1) - gt.unsqueeze(0)
        assert gt_diff.shape[0] == gt.shape[0] and gt_diff.shape[1] == gt.shape[0], f"invalid pred diff shape {pred_diff.shape} or gt diff shape {gt_diff.shape}"
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.alpha)
        loss = loss.mean().div(2)
        return loss

class ClippedL1Loss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, gt):
        loss = self.l1(pred, gt)
        mask = (loss - self.tau) > 0
        loss = torch.mean(mask * loss)
        return loss


if __name__ == "__main__":
    with torch.no_grad():
        shape = (2, 10)
        x, y = torch.rand(*shape), torch.rand(*shape)
        loss = CCC_Rdrop_Loss()
        print(loss(x, {'labels': y})['loss'])
