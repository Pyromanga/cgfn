import torch


def svd_signature(states):
    X = states.mean(dim=1)
        X = X - X.mean(dim=0, keepdim=True)
            U, S, _ = torch.linalg.svd(X, full_matrices=False)
                return U * S, S


                def steering_loss(states, target_sig, basis):
                    X = states.mean(dim=1)
                        X = X - X.mean(dim=0, keepdim=True)
                            sig = X @ basis.T
                                return ((sig - target_sig) ** 2).mean()