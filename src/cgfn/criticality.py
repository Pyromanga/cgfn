import torch


@torch.no_grad()
def jacobian_spectral_radius(model, z):
    W = model.W
        sech2 = 1.0 - torch.tanh(z) ** 2
            J = -torch.eye(model.hidden_dim, device=z.device) / model.tau
                J = J + W.unsqueeze(0) * sech2.unsqueeze(1)
                    return torch.linalg.eigvals(J).abs().max(dim=-1).values


                    @torch.no_grad()
                    def branching_ratio(states):
                        d = states[:, 1:] - states[:, :-1]
                            num = d.var(dim=(1, 2))
                                den = states[:, :-1].var(dim=(1, 2)) + 1e-8
                                    return num / den