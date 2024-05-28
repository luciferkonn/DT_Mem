from typing import Mapping, Optional, Tuple
import torch.nn.functional as F
import torch


def encode_return(ret: torch.Tensor, ret_range: Tuple[int]) -> torch.Tensor:
    """Encode return values into discrete return tokens."""
    ret = ret.to(dtype=torch.int32)
    ret = torch.clip(ret, ret_range[0], ret_range[1])
    ret = ret-ret_range[0]
    return ret


def encode_reward(rew: torch.Tensor) -> torch.Tensor:
    """
    Encode reward values
    # 0: no reward 1: positive reward 2: terminal reward 3: negative reward
    """
    rew = (rew > 0)*1+(rew < 0)*3
    return rew.to(dtype=torch.int32)


def decode_return(ret: torch.Tensor, ret_range: Tuple[int]) -> torch.Tensor:
    ret = ret.to(dtype=torch.int32)
    ret = ret + ret_range[0]
    return ret


def cross_entropy(logits, labels):
    """Applies sparse cross entropy loss between logits and target labels"""
    # labels = F.one_hot(labels.to(dtype=torch.int64),
    #                    logits.shape[-1]).squeeze(2)
    # loss = -labels * F.log_softmax(logits)
    # return torch.mean(loss)
    N, T = labels.size(0), labels.size(1)
    labels = labels.reshape(N*T,).to(dtype=torch.int64)
    logits = logits.reshape(N*T, -1)
    loss = F.cross_entropy(logits, labels)
    return loss


def accuracy(logits, labels):
    predicted_label = torch.argmax(logits, -1)
    acc = torch.eq(predicted_label, labels.squeeze(-1)).to(dtype=torch.float32)
    # print(f"predicted {predicted_label.cpu().tolist()}")
    # print(f"target {labels.squeeze(-1).cpu().tolist()}")
    return torch.mean(acc)


def sample_from_logits(
    logits: torch.Tensor,
    deterministic: Optional[bool] = False,
    temperature: Optional[float] = 1e0,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None
):
    if deterministic:
        sample = torch.argmax(logits, dim=-1)
    else:
        if top_percentile is not None:
            percentile = torch.quantile(logits, top_percentile/100, dim=-1)
            logits = torch.where(
                logits > percentile.unsqueeze(-1), logits, -torch.inf)
        if top_k is not None:
            logits, top_indices = torch.topk(logits, top_k, dim=-1)
        dist = torch.distributions.Categorical(logits=(logits*temperature))
        sample = dist.sample()
        if top_k is not None:
            sample_shape = sample.shape
            top_indices = top_indices.reshape(-1, top_k)
            sample = sample.flatten()
            sample = top_indices[torch.arange(len(sample)), sample]
    return sample


def autoregressive_generate(
    inputs: Mapping[str, torch.Tensor],
    logits_fn,
    name: str,
    seq_len: int,
    deterministic: Optional[bool] = False,
    temperature: Optional[float] = 1e+0,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    val = torch.zeros_like(inputs[name])

    for t in range(0, seq_len):
        datapoint = dict(inputs)
        datapoint[name] = val
        logits = logits_fn(datapoint)
        sample = sample_from_logits(logits, deterministic=deterministic,
                                    temperature=temperature, top_k=top_k,
                                    top_percentile=top_percentile)
        val[:, t] = sample
    return val
