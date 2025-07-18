import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    """Restricted Boltzmann Machine with binary visible and hidden units.

    Uses CD-k (default k=1) for training and provides utilities for sampling and
    converting the energy into a QUBO (biases + pairwise couplings)."""

    def __init__(self, n_visible: int, n_hidden: int, k: int = 1, lr: float = 1e-3, device: str = "cpu"):
        super().__init__()
        self.n_visible = n_visible
        self.visible_size = n_visible  # Alias for compatibility
        self.n_hidden = n_hidden
        self.hidden_size = n_hidden  # Alias for compatibility
        self.k = k
        self.lr = lr
        self.device = device

        # Parameters: weights and biases
        # Initialize with small random values.
        self.W = nn.Parameter(0.01 * torch.randn(n_visible, n_hidden, device=device))
        self.v_bias = nn.Parameter(torch.zeros(n_visible, device=device))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden, device=device))

    def _sample_from_p(self, p: torch.Tensor) -> torch.Tensor:
        """Sample binary states given Bernoulli probabilities."""
        # Clamp probabilities to avoid numerical instability with bernoulli
        p_clamped = torch.clamp(p, 1e-8, 1.0 - 1e-8)
        return torch.bernoulli(p_clamped)

    def _v_to_h(self, v: torch.Tensor) -> torch.Tensor:
        p_h = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        return self._sample_from_p(p_h), p_h

    def _h_to_v(self, h: torch.Tensor) -> torch.Tensor:
        p_v = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        return self._sample_from_p(p_v), p_v

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """One Gibbs sampling step starting from visible v."""
        h, _ = self._v_to_h(v)
        v_neg, _ = self._h_to_v(h)
        return v_neg

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Compute free energy of visible vectors v."""
        vbias_term = v @ self.v_bias
        hidden_term = torch.sum(F.softplus(F.linear(v, self.W.t(), self.h_bias)), dim=1)
        return -vbias_term - hidden_term

    def cd_loss(self, v0: torch.Tensor) -> torch.Tensor:
        """Compute CD-k loss for a batch of visible vectors."""
        vk = v0
        for _ in range(self.k):
            hk, _ = self._v_to_h(vk)
            vk, _ = self._h_to_v(hk)
        return torch.mean(self.free_energy(v0)) - torch.mean(self.free_energy(vk.detach()))

    def train_step(self, v: torch.Tensor, optimizer: torch.optim.Optimizer, k: int = None):
        """
        Train RBM with contrastive divergence.

        Args:
            v: Visible units input
            optimizer: Optimizer for RBM parameters
            k: Number of CD steps (overrides self.k if provided)

        Returns:
            Loss value
        """
        if k is None:
            k = self.k

        optimizer.zero_grad()

        # Compute CD-k loss with specified k
        vk = v
        for _ in range(k):
            hk, _ = self._v_to_h(vk)
            vk, _ = self._h_to_v(hk)

        loss = torch.mean(self.free_energy(v)) - torch.mean(self.free_energy(vk.detach()))
        loss.backward()
        optimizer.step()
        return loss.item()

    # -------------------------------------------------
    # QUBO conversion utilities
    # -------------------------------------------------
    def qubo_terms(self):
        """Return bias vector h and coupling matrix J representing the RBM energy
        E(v, h) = -v^T W h - v_bias^T v - h_bias^T h
        as a QUBO in variables [v || h].
        
        QUBO form: minimize x^T J x + h^T x where x = [v, h]
        We want: x^T J x + h^T x = -v^T W h - v_bias^T v - h_bias^T h
        """
        # Concatenate variables: first visible, then hidden.
        n_total = self.n_visible + self.n_hidden
        J = torch.zeros(n_total, n_total, device=self.device)
        h_vec = torch.zeros(n_total, device=self.device)

        # Linear terms: h^T x = h_vec^T [v, h] = -v_bias^T v - h_bias^T h
        h_vec[: self.n_visible] = -self.v_bias.detach()
        h_vec[self.n_visible :] = -self.h_bias.detach()

        # Quadratic terms: x^T J x should give us -v^T W h
        # Since x^T J x = [v, h]^T J [v, h], and we want -v^T W h,
        # we need J[v_idx, h_idx] = -W[i,j]/2 and J[h_idx, v_idx] = -W[i,j]/2
        # so that v_i * J[v_idx, h_idx] * h_j + h_j * J[h_idx, v_idx] * v_i = -W[i,j] * v_i * h_j
        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                J[i, self.n_visible + j] = -self.W[i, j].detach() / 2.0
                J[self.n_visible + j, i] = -self.W[i, j].detach() / 2.0

        return J, h_vec

    # -------------------------------------------------
    # Approximate marginalisation over hidden units (TAP 2nd-order) to obtain
    # an effective pairwise energy purely over visible units. This is useful
    # for constructing a QUBO that does *not* include hidden variables.
    # -------------------------------------------------
    def visible_pairwise_approx(self):
        """Return (J_vis, h_vis) over only visible bits via a 2nd-order expansion.

        F(v) = -aᵀv - ∑_j log(1 + exp(b_j + W_{:,j}ᵀ v))

        Using a second-order Taylor expansion around v=0.5, the pairwise term is
            J_{ik} = -½ ∑_j σ(b_j)(1-σ(b_j)) W_{ij} W_{kj}
        and the linear term is
            h_i  = -a_i - ½ ∑_j σ(b_j) W_{ij}
        """
        with torch.no_grad():
            s = torch.sigmoid(self.h_bias)  # (n_hidden,)
            # (n_visible, n_hidden) * (n_hidden,)  -> (n_visible, n_hidden)
            Ws = self.W * s.unsqueeze(0)
            # Linear biases
            h_vis = -self.v_bias - 0.5 * Ws.sum(dim=1)

            # Pairwise couplings
            coeff = (s * (1.0 - s))  # (n_hidden,)
            W_scaled = self.W * coeff.unsqueeze(0)
            J_vis = -0.5 * torch.matmul(W_scaled, self.W.t())  # (n_visible, n_visible)
            # Ensure symmetry & zero diagonal
            J_vis = 0.5 * (J_vis + J_vis.t())
            J_vis.fill_diagonal_(0.0)
        return J_vis, h_vis 