
import torch


with torch.no_grad():



def cal_time_p_and_p_model(self, y_unlb, time_p, p_model, label_hist):
    prob_w = torch.softmax(y_unlb, dim=1)
    max_probs, max_idx = torch.max(prob_w, dim=-1)
    if time_p is None:
        time_p = max_probs.mean()
    else:
        time_p = time_p * 0.999 + max_probs.mean() * 0.001
    if p_model is None:
        p_model = torch.mean(prob_w, dim=0)
    else:  #
        p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
    if label_hist is None:
        label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(
            p_model.dtype)
        label_hist = label_hist / label_hist.sum()
    else:  
        hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(
            p_model.dtype)
        label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
    return time_p, p_model, label_hist
