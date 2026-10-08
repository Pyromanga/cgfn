import torch


def mean_state_signature(states):
    """states: (B, T, H) -> (H,) gemittelt ueber Batch und letzte 25% der Zeit."""
    T = states.shape[1]
    tail = states[:, int(0.75 * T):, :]
    return tail.mean(dim=(0, 1))


def covariance_signature(states):
    """states: (B, T, H) -> (H, H) Kovarianz ueber Batch*Zeit."""
    B, T, H = states.shape
    X = states.reshape(B * T, H)
    X = X - X.mean(dim=0, keepdim=True)
    return (X.T @ X) / (X.shape[0] - 1)


def signature_distance(sig_a, sig_b):
    return (sig_a - sig_b).norm().item()


def steering_loss(states, target_sig):
    sig = mean_state_signature(states)
    return ((sig - target_sig) ** 2).mean()
