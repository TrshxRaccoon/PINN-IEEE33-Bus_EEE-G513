#!/usr/bin/env python3
"""
models.py -- Graph Neural Network Architectures for Physics-Informed Power System State Estimation
Target System: IEEE 33-Bus Radial Distribution Feeder

Implements:
  1. EdgeAttributedConv: PyTorch Geometric MessagePassing layer with edge conditioning:
       m_ji = Linear_v(h_j) * act(Linear_e(edge_attr_ji))
       m_ji_norm = D^{-1/2} A D^{-1/2} * m_ji (symmetric degree normalization)
       agg_i = sum_{j in N(i)} m_ji_norm
       h_i^(l+1) = act(Linear_update(h_i^(l) + agg_i)) (residual skip connection)
  2. EdgeAttributedGCN: Multi-layer edge-attributed GCN mapping 6 node features to 2 states (norm_vm, norm_va)
  3. AC Power Flow Residual & Physical Unit Reconstruction helpers (complex128 / float64)
"""

from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# Enforce double precision default
torch.set_default_dtype(torch.float64)


# ==============================================================================
# 1. EDGE-ATTRIBUTED MESSAGE PASSING LAYER
# ==============================================================================
class EdgeAttributedConv(MessagePassing):
    """
    Edge-conditioned message passing layer for power system graphs.

    Computes:
      m_ji = Linear_v(h_j) * act(Linear_e(edge_attr_ji))
      Degree normalization: D^{-1/2} A D^{-1/2} symmetric normalization.
      Aggregation: sum of incoming messages.
      Update with residual skip connection:
        h_i^(l+1) = act(Linear_update(h_i^(l) + agg_i))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int = 4,
        act: str = "tanh",
    ):
        super().__init__(aggr="add", flow="source_to_target")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.act_name = act.lower()

        # Activation function
        if self.act_name == "tanh":
            self.act = nn.Tanh()
        elif self.act_name == "silu":
            self.act = nn.SiLU()
        elif self.act_name == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(
                f"Unsupported activation: '{act}'. Expected 'tanh', 'silu', or 'relu'."
            )

        # Linear transformations (strictly float64)
        self.lin_v = nn.Linear(in_channels, out_channels, bias=True, dtype=torch.float64)
        self.lin_e = nn.Linear(edge_channels, out_channels, bias=True, dtype=torch.float64)
        self.lin_update = nn.Linear(out_channels, out_channels, bias=True, dtype=torch.float64)

        # Residual skip connection projection if channel dimensions differ
        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels, bias=False, dtype=torch.float64)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward step of edge-attributed convolution.

        Args:
          x: Node features of shape (N, in_channels) in float64.
          edge_index: Graph connectivity (2, E) in long.
          edge_attr: Edge attributes (E, edge_channels) in float64.
          edge_weight: Optional precomputed symmetric normalization weights (E,).
        """
        if x.dtype != torch.float64:
            x = x.to(torch.float64)
        if edge_attr.dtype != torch.float64:
            edge_attr = edge_attr.to(torch.float64)

        # Symmetric degree normalization: D^{-1/2} A D^{-1/2}
        # Note: self-loops are already present in edge_index from Ybus diagonal
        if edge_weight is None:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                num_nodes=x.size(0),
                add_self_loops=False,
                dtype=x.dtype,
            )

        # Project source node features
        h_v = self.lin_v(x)

        # Propagate messages from j to i
        agg = self.propagate(
            edge_index,
            h_v=h_v,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            size=(x.size(0), x.size(0)),
        )

        # Residual skip connection: h_i^(l+1) = act(Linear_update(h_i^(l) + agg_i))
        h_res = self.res_proj(x)
        out = self.act(self.lin_update(h_res + agg))
        return out

    def message(
        self,
        h_v_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        m_ji = Linear_v(h_j) * act(Linear_e(edge_attr_ji))
        Normalized by symmetric degree weight D^{-1/2} A D^{-1/2}.
        """
        # Edge modulation: act(Linear_e(edge_attr_ji))
        e_msg = self.act(self.lin_e(edge_attr))
        msg = h_v_j * e_msg
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg


# ==============================================================================
# 2. EDGE-ATTRIBUTED GCN ARCHITECTURE
# ==============================================================================
class EdgeAttributedGCN(nn.Module):
    """
    Edge-Attributed Graph Convolutional Network for Distribution Grid State Estimation.

    Inputs:
      - Node features: x of shape (N, 6) in float64:
          [norm_vm, norm_va, norm_p, norm_q, mask, is_slack]
      - Edge attributes: edge_attr of shape (E, 4) in float64:
          [G, B, |Y|, angle]
      - Edge connectivity: edge_index of shape (2, E) in long:
          (64 directed line edges + 33 self-loops = 97 edges for IEEE 33)

    Outputs:
      - Per-node state predictions [norm_vm, norm_va] for each bus:
          shape: (batch_size, num_nodes, 2) when reshape_batch=True (default)
          or (N_nodes, 2) when reshape_batch=False.
    """

    def __init__(
        self,
        in_channels: int = 6,
        edge_channels: int = 4,
        out_channels: int = 2,
        hidden_dims: Optional[List[int]] = None,
        act: str = "tanh",
        num_nodes: int = 33,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
        self.hidden_dims = list(hidden_dims)
        self.act_name = act.lower()
        self.num_nodes = num_nodes

        # Message passing layers
        self.convs = nn.ModuleList()
        prev_dim = in_channels
        for h_dim in self.hidden_dims:
            self.convs.append(
                EdgeAttributedConv(
                    in_channels=prev_dim,
                    out_channels=h_dim,
                    edge_channels=edge_channels,
                    act=act,
                )
            )
            prev_dim = h_dim

        # Final projection layer: Linear(hidden, out_channels=2)
        self.head = nn.Linear(prev_dim, out_channels, bias=True, dtype=torch.float64)

        # Strictly operate in float64
        self.to(torch.float64)

    def forward(
        self,
        data_or_x: Union[torch.Tensor, Data, Batch],
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        reshape_batch: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        Accepts:
          - A single Data or batched Batch object from torch_geometric.loader.DataLoader:
              model(batch)
          - Explicit tensors:
              model(x, edge_index, edge_attr, batch=None)

        Returns:
          Tensor of shape (batch_size, num_nodes, 2) if reshape_batch is True
          and total nodes is a multiple of num_nodes, else shape (N_total, 2).
        """
        # Unpack Data / Batch or tensor arguments
        if hasattr(data_or_x, "x") and hasattr(data_or_x, "edge_index"):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
            edge_attr = getattr(data_or_x, "edge_attr", None)
            batch = getattr(data_or_x, "batch", None)
        else:
            x = data_or_x

        if edge_index is None:
            raise ValueError("edge_index must be provided.")
        if edge_attr is None:
            raise ValueError("edge_attr must be provided.")

        if x.dtype != torch.float64:
            x = x.to(torch.float64)
        if edge_attr.dtype != torch.float64:
            edge_attr = edge_attr.to(torch.float64)

        # Message passing pipeline
        h = x
        for conv in self.convs:
            h = conv(h, edge_index=edge_index, edge_attr=edge_attr)

        # Projection to 2 values per node: [norm_vm, norm_va]
        out = self.head(h)

        # Automatic reshaping for downstream evaluation / physics loss
        if reshape_batch:
            total_nodes = out.size(0)
            if total_nodes % self.num_nodes == 0:
                bs = total_nodes // self.num_nodes
                out = out.view(bs, self.num_nodes, self.out_channels)

        return out

    def forward_nodes(
        self,
        data_or_x: Union[torch.Tensor, Data, Batch],
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method returning unshaped (N_nodes, 2) node predictions."""
        return self.forward(
            data_or_x=data_or_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            reshape_batch=False,
        )


# ==============================================================================
# 3. PHYSICS & EVALUATION UTILITIES (Section 3 & Section 7 Compliance)
# ==============================================================================
def power_residual(
    vm: torch.Tensor,
    va: torch.Tensor,
    Y: torch.Tensor,
    p_ps: torch.Tensor,
    q_ps: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """
    AC Power Flow Residual Calculation in Double Precision (complex128 / float64):
      V = polar(vm, va) (complex128)
      S = V * conj(V @ Y.T) (calculated complex bus injections)
      rP = S.real - p_ps
      rQ = S.imag - q_ps
      loss = mean(rP[:, keep]**2 + rQ[:, keep]**2)
    """
    if vm.dim() == 1:
        vm = vm.unsqueeze(0)
    if va.dim() == 1:
        va = va.unsqueeze(0)
    if p_ps.dim() == 1:
        p_ps = p_ps.unsqueeze(0)
    if q_ps.dim() == 1:
        q_ps = q_ps.unsqueeze(0)

    if Y.dtype != torch.complex128:
        Y = Y.to(torch.complex128)
    if vm.dtype != torch.float64:
        vm = vm.to(torch.float64)
    if va.dtype != torch.float64:
        va = va.to(torch.float64)

    V = torch.polar(vm, va)
    S = V * torch.conj(V @ Y.T)
    rP = S.real - p_ps
    rQ = S.imag - q_ps
    return (rP[:, keep] ** 2 + rQ[:, keep] ** 2).mean()


def to_physical(
    y_norm: torch.Tensor,
    scaler: Any,
    device: Optional[torch.device] = None,
    num_nodes: int = 33,
    slack_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert standardized model outputs back to physical units and overwrite slack bus.

    Accepts y_norm of shape:
      - (B, num_nodes, 2)
      - (B, 2 * num_nodes) [MLP flat format]
      - (num_nodes, 2)
    Returns:
      vm: (B, num_nodes) in p.u. with slack set to 1.0 pu
      va: (B, num_nodes) in rad with slack set to 0.0 rad
    """
    if device is None:
        device = y_norm.device

    t_vm_mean = torch.as_tensor(scaler.target_vm_mean, dtype=torch.float64, device=device)
    t_vm_std = torch.as_tensor(scaler.target_vm_std, dtype=torch.float64, device=device)
    t_va_mean = torch.as_tensor(scaler.target_va_mean, dtype=torch.float64, device=device)
    t_va_std = torch.as_tensor(scaler.target_va_std, dtype=torch.float64, device=device)

    if y_norm.dim() == 3 and y_norm.size(1) == num_nodes and y_norm.size(2) == 2:
        norm_vm = y_norm[:, :, 0]
        norm_va = y_norm[:, :, 1]
    elif y_norm.dim() == 2 and y_norm.size(0) == num_nodes and y_norm.size(1) == 2:
        norm_vm = y_norm[:, 0].unsqueeze(0)
        norm_va = y_norm[:, 1].unsqueeze(0)
    elif y_norm.dim() == 2 and y_norm.size(1) == 2 * num_nodes:
        norm_vm = y_norm[:, :num_nodes]
        norm_va = y_norm[:, num_nodes:]
    else:
        raise ValueError(f"Unrecognized y_norm shape: {y_norm.shape}")

    vm = norm_vm * t_vm_std + t_vm_mean
    va = norm_va * t_va_std + t_va_mean

    # Overwrite slack bus with known physical reference (1.0 pu, 0.0 rad)
    vm = vm.clone()
    va = va.clone()
    vm[:, slack_idx] = 1.0
    va[:, slack_idx] = 0.0

    return vm, va
