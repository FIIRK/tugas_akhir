import torch
from torch import nn
import torch.nn.functional as F

### Loss Metrics    
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE/2

### Accuracy Metrics
def pixel_accuracy(predictions, targets):
  with torch.no_grad():
    predictions = torch.round(torch.sigmoid(predictions))
    pixel_accuracy = torch.sum(predictions == targets) / targets.numel()
    return pixel_accuracy

def F1_score(output, label):
  with torch.no_grad():
    result = torch.round(torch.sigmoid(output))
    tp = result * label
    prc = tp.sum()/result.sum()
    rec = tp.sum()/label.sum()
    return 2*prc*rec/(prc+rec+1e-6)
