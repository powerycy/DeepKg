import torch
def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    return torch.sum(y_true * y_pred),torch.sum(y_true + y_pred)