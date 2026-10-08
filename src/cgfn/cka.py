import torch


def center(K):
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def cka(X, Y, kernel="linear"):
    """X: (n, d1), Y: (n, d2). Rueckgabe: Skalar in [0, 1]."""
    if kernel == "linear":
        Kx = X @ X.T
        Ky = Y @ Y.T
    elif kernel == "rbf":
        def rbf(A, sigma=None):
            d2 = torch.cdist(A, A) ** 2
            if sigma is None:
                sigma = d2[d2 > 0].median().sqrt()
            return torch.exp(-d2 / (2 * sigma ** 2))
        Kx = rbf(X)
        Ky = rbf(Y)
    else:
        raise ValueError(kernel)
    Kxc = center(Kx)
    Kyc = center(Ky)
    num = (Kxc * Kyc).sum()
    den = (Kxc * Kxc).sum().sqrt() * (Kyc * Kyc).sum().sqrt()
    return (num / (den + 1e-12)).item()
