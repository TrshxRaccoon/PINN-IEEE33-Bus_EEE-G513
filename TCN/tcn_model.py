#!/usr/bin/env python3
"""
tcn_model.py -- Temporal Convolutional Network Architectures for Power System State Estimation
Physics-Informed State Estimation for Distribution Grids under PMU Dropout.

TARGET SYSTEM:
    IEEE 33-bus radial distribution feeder (pandapower `case33bw`, 12.66 kV, 10 MVA base).

KEY ARCHITECTURAL PROPERTIES:
    1. Pure PyTorch Temporal Convolutional Network (TCN) in `torch.float64`.
    2. Dilated Causal 1D Convolutions:
       - Conv1d with kernel size k, dilation d, padding (k - 1) * d.
       - Chomp1d right-trimming to ensure strict causality (no future leakage).
       - Dilations grow exponentially: d = 2^0, 2^1, 2^2, 2^3, ...
    3. Residual Temporal Blocks:
       - Two Conv1d-Chomp-Act-Dropout stages per block.
       - Residual skip connection: 1x1 Conv1d if in_channels != out_channels, else Identity.
       - Configurable activations: Tanh (default), SiLU, ReLU.
       - Weight normalization support with multi-version PyTorch compatibility.
    4. State Estimator Head:
       - Maps sequential history of 114 features across T time steps:
           Block 1: 16 Standardized PMU |V| * mask
           Block 2: 16 Standardized PMU theta * mask
           Block 3: 33 Standardized P_pseudo injections
           Block 4: 33 Standardized Q_pseudo injections
           Block 5: 16 Raw PMU binary mask
       - Extracts last time-step feature vector h_T in R^{B x C_last}.
       - Linear projection head: h_T -> 66 outputs [|V|_{0..32}, theta_{0..32}].
       - Enforces double precision (torch.float64) throughout all layers and tensors.
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

# Enforce double precision globally by default
torch.set_default_dtype(torch.float64)


# ==============================================================================
# 0. UTILITY HELPERS: ACTIVATION & WEIGHT NORMALIZATION
# ==============================================================================
def _get_activation(act: Union[str, nn.Module]) -> nn.Module:
    """Resolve activation function from string or module instance."""
    if isinstance(act, nn.Module):
        return act
    act_name = str(act).strip().lower()
    if act_name in ("tanh", "nn.tanh()", "nn.tanh"):
        return nn.Tanh()
    elif act_name in ("silu", "swish", "nn.silu()", "nn.silu"):
        return nn.SiLU()
    elif act_name in ("relu", "nn.relu()", "nn.relu"):
        return nn.ReLU()
    elif act_name in ("gelu", "nn.gelu()", "nn.gelu"):
        return nn.GELU()
    elif act_name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(
            f"Unsupported activation: '{act}'. Expected 'tanh', 'silu', or 'relu'."
        )


def _apply_weight_norm(module: nn.Module) -> nn.Module:
    """
    Apply weight normalization to a module with multi-version PyTorch compatibility:
    - PyTorch >= 2.1: torch.nn.utils.parametrizations.weight_norm
    - Older PyTorch:  torch.nn.utils.weight_norm
    """
    try:
        from torch.nn.utils.parametrizations import weight_norm
        return weight_norm(module)
    except (ImportError, AttributeError):
        from torch.nn.utils import weight_norm
        return weight_norm(module)


# ==============================================================================
# 1. CAUSAL CHOMP LAYER
# ==============================================================================
class Chomp1d(nn.Module):
    """
    Trims the trailing (right-hand) padding from a 1D convolution to enforce strict causality.
    For an input tensor of shape (B, C, T + chomp_size), returns (B, C, T).
    Safely handles chomp_size == 0 to avoid zero-length slice errors.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

    def extra_repr(self) -> str:
        return f"chomp_size={self.chomp_size}"


# ==============================================================================
# 2. TEMPORAL RESIDUAL BLOCK
# ==============================================================================
class TemporalBlock(nn.Module):
    """
    Causal Dilated Temporal Residual Block (Bai, Kolter, Koltun 2018).

    Architecture:
      Conv1d(in, out, k, dilation=d, padding=(k-1)*d)
      -> Chomp1d((k-1)*d)
      -> Activation
      -> Dropout(p)
      -> Conv1d(out, out, k, dilation=d, padding=(k-1)*d)
      -> Chomp1d((k-1)*d)
      -> Activation
      -> Dropout(p)
      + Skip connection: 1x1 Conv1d if in != out, else Identity
      -> Final Activation

    All internal weights and calculations strictly operate in `torch.float64`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
        act: Union[str, nn.Module] = "tanh",
        use_weight_norm: bool = True,
        apply_final_act: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_p = dropout
        self.apply_final_act = apply_final_act

        # First dilated convolution stage
        conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            dtype=torch.float64,
        )
        if use_weight_norm:
            conv1 = _apply_weight_norm(conv1)
        chomp1 = Chomp1d(padding)
        act1 = _get_activation(act)
        drop1 = nn.Dropout(dropout)

        # Second dilated convolution stage
        conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            dtype=torch.float64,
        )
        if use_weight_norm:
            conv2 = _apply_weight_norm(conv2)
        chomp2 = Chomp1d(padding)
        act2 = _get_activation(act)
        drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            conv1, chomp1, act1, drop1,
            conv2, chomp2, act2, drop2
        )

        # Residual projection if channel count changes
        if in_channels != out_channels:
            res_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
                dtype=torch.float64,
            )
            if use_weight_norm:
                res_conv = _apply_weight_norm(res_conv)
            self.downsample = res_conv
        else:
            self.downsample = nn.Identity()

        # Final non-linear activation after residual addition
        if apply_final_act:
            self.final_act = _get_activation(act)
        else:
            self.final_act = nn.Identity()

        # Ensure double precision
        self.to(torch.float64)
        self.init_weights()

    def init_weights(self):
        """Initialize convolution kernels with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
          x: Input tensor of shape (B, in_channels, T) in float64.
        Returns:
          out: Output tensor of shape (B, out_channels, T) in float64.
        """
        if x.dtype != torch.float64:
            x = x.to(torch.float64)
        out = self.net(x)
        res = self.downsample(x)
        return self.final_act(out + res)


# ==============================================================================
# 3. TEMPORAL CONVOLUTIONAL NETWORK BACKBONE
# ==============================================================================
class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network backbone.
    Stacks TemporalBlocks with exponentially increasing dilations:
      d = 2^0, 2^1, 2^2, ..., 2^(L-1)
    Maintains strict causality, constant sequence length T, and double precision (torch.float64).
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        act: Union[str, nn.Module] = "tanh",
        use_weight_norm: bool = True,
        apply_final_act: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.num_levels = len(num_channels)

        layers = []
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    act=act,
                    use_weight_norm=use_weight_norm,
                    apply_final_act=apply_final_act,
                )
            )

        self.network = nn.Sequential(*layers)
        self.to(torch.float64)

    @property
    def receptive_field(self) -> int:
        """
        Calculate the total theoretical receptive field (in time steps):
        RF = 1 + sum_{i=0}^{L-1} 2 * (k - 1) * 2^i = 1 + 2 * (k - 1) * (2^L - 1)
        """
        return 1 + 2 * (self.kernel_size - 1) * ((2 ** self.num_levels) - 1)

    def get_receptive_field(self) -> int:
        return self.receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
          x: Input tensor of shape (B, in_channels, T) in float64.
        Returns:
          out: Output tensor of shape (B, num_channels[-1], T) in float64.
        """
        if x.dtype != torch.float64:
            x = x.to(torch.float64)
        return self.network(x)


# ==============================================================================
# 4. STATE ESTIMATOR TCN FOR IEEE 33-BUS
# ==============================================================================
class StateEstimatorTCN(nn.Module):
    """
    Temporal Convolutional Network State Estimator for Distribution Grids.

    Accepts sequential telemetry input across T historical time steps:
      x: Shape (B, C_in, T) or (B, T, C_in), where C_in = 114 features:
         - [0:16]   : Standardized PMU Voltage Magnitudes (|V|) * mask
         - [16:32]  : Standardized PMU Voltage Angles (theta) * mask
         - [32:65]  : Standardized P pseudo-measurements (33 buses)
         - [65:98]  : Standardized Q pseudo-measurements (33 buses)
         - [98:114] : Raw binary PMU availability mask (16 PMUs)
    
    Processes the sequence through a causal dilated TemporalConvNet, extracts
    the final time-step representation h_T in R^{B x C_last}, and projects through
    a linear prediction head to predict the full 66-dimensional network state:
      y: Shape (B, 66) representing [|V|_{0..32}, theta_{0..32}].

    Strictly implemented in double precision (`torch.float64`).
    """

    def __init__(
        self,
        in_channels: int = 114,
        out_channels: int = 66,
        hidden_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        act: Union[str, nn.Module] = "tanh",
        use_weight_norm: bool = True,
        apply_final_act: bool = True,
    ):
        super().__init__()
        if hidden_channels is None:
            # Default 4-layer architecture with 64 channels:
            # Receptive field = 1 + 2 * (3-1) * (2^4 - 1) = 1 + 4 * 15 = 61 steps
            hidden_channels = [64, 64, 64, 64]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = list(hidden_channels)
        self.kernel_size = kernel_size
        self.dropout_p = dropout
        self.act_name = str(act)

        # Causal Temporal Convolutional Network backbone
        self.tcn = TemporalConvNet(
            in_channels=in_channels,
            num_channels=self.hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            act=act,
            use_weight_norm=use_weight_norm,
            apply_final_act=apply_final_act,
        )

        # Linear prediction head: C_last -> out_channels (66)
        c_last = self.hidden_channels[-1]
        self.head = nn.Linear(c_last, out_channels, bias=True, dtype=torch.float64)

        # Enforce float64 throughout
        self.to(torch.float64)

    @property
    def receptive_field(self) -> int:
        """Theoretical receptive field in time steps."""
        return self.tcn.receptive_field

    def get_receptive_field(self) -> int:
        return self.receptive_field

    def forward(
        self,
        x: torch.Tensor,
        return_sequences: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
          x: Input tensor of shape (B, C_in, T) or (B, T, C_in) or snapshot (B, C_in).
             Default C_in = 114.
          return_sequences: If True, returns full sequence of predictions (B, T, 66).
                            If False (default), returns prediction at last time step (B, 66).

        Returns:
          y_pred: Predicted state vector in double precision (torch.float64).
                  Shape (B, 66) if return_sequences=False,
                  Shape (B, T, 66) if return_sequences=True.
        """
        if x.dtype != torch.float64:
            x = x.to(torch.float64)

        # Handle 2D snapshot input: (B, C_in) -> (B, C_in, 1)
        if x.dim() == 2:
            if x.shape[1] == self.in_channels:
                x_conv = x.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Expected 2D input with second dim={self.in_channels}, but got shape {tuple(x.shape)}"
                )
        elif x.dim() == 3:
            # Check orientation: Conv1d requires (B, C_in, T)
            if x.shape[1] == self.in_channels:
                x_conv = x
            elif x.shape[2] == self.in_channels:
                # Transpose (B, T, C_in) -> (B, C_in, T)
                x_conv = x.transpose(1, 2)
            else:
                raise ValueError(
                    f"Input tensor has shape {tuple(x.shape)}, neither dim 1 nor dim 2 "
                    f"matches expected in_channels={self.in_channels}."
                )
        else:
            raise ValueError(
                f"Expected 2D or 3D input tensor, but got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )

        # Feature extraction through dilated causal conv net -> (B, C_last, T)
        feat = self.tcn(x_conv)

        if return_sequences:
            # Map all time steps: (B, C_last, T) -> (B, T, C_last) -> (B, T, out_channels)
            feat_seq = feat.transpose(1, 2)
            out = self.head(feat_seq)
            return out
        else:
            # Extract the final time-step representation h_T -> (B, C_last)
            h_T = feat[:, :, -1]
            out = self.head(h_T)
            return out


# ==============================================================================
# 5. PHYSICAL CONVERSIONS & AC POWER FLOW RESIDUAL HELPERS
# ==============================================================================
def to_physical(
    y_pred: torch.Tensor,
    target_vm_mean: Union[np.ndarray, torch.Tensor],
    target_vm_std: Union[np.ndarray, torch.Tensor],
    target_va_mean: Union[np.ndarray, torch.Tensor],
    target_va_std: Union[np.ndarray, torch.Tensor],
    slack_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    De-standardize neural network output to physical units:
      vm = norm_vm * std_vm + mean_vm  (pu)
      va = norm_va * std_va + mean_va  (rad)
    And explicitly enforces exact physical constants at the slack bus:
      vm[:, slack] = 1.0 pu
      va[:, slack] = 0.0 rad

    Args:
      y_pred: standardized output tensor of shape (B, 66) or (B, T, 66).
      target_vm_mean, target_vm_std: (33,) statistics for voltage magnitude.
      target_va_mean, target_va_std: (33,) statistics for voltage phase angle.
      slack_idx: index of slack bus (default 0).

    Returns:
      vm: voltage magnitude in pu (B, 33) or (B, T, 33).
      va: voltage phase angle in rad (B, 33) or (B, T, 33).
    """
    device = y_pred.device
    tvm_m = torch.as_tensor(target_vm_mean, dtype=torch.float64, device=device)
    tvm_s = torch.as_tensor(target_vm_std, dtype=torch.float64, device=device)
    tva_m = torch.as_tensor(target_va_mean, dtype=torch.float64, device=device)
    tva_s = torch.as_tensor(target_va_std, dtype=torch.float64, device=device)

    # Split into vm (first 33) and va (last 33)
    norm_vm = y_pred[..., :33]
    norm_va = y_pred[..., 33:66]

    # De-standardize
    vm = norm_vm * tvm_s + tvm_m
    va = norm_va * tva_s + tva_m

    # Enforce known slack bus boundary condition
    vm = vm.clone()
    va = va.clone()
    vm[..., slack_idx] = 1.0
    va[..., slack_idx] = 0.0

    return vm, va


def power_residual(
    vm: torch.Tensor,
    va: torch.Tensor,
    Y: torch.Tensor,
    p_ps: torch.Tensor,
    q_ps: torch.Tensor,
    keep_buses: Optional[Union[torch.Tensor, Sequence[int]]] = None,
) -> torch.Tensor:
    """
    Computes AC power flow injection residual loss (float64 and complex128):
      V = vm * exp(1j * va)
      S_calc = V * conj(V @ Y.T)
      rP = S_calc.real - p_ps
      rQ = S_calc.imag - q_ps
      L_phys = mean(rP[:, keep_buses]^2 + rQ[:, keep_buses]^2)

    Args:
      vm: voltage magnitude in pu, shape (B, 33)
      va: voltage phase angle in rad, shape (B, 33)
      Y: complex bus admittance matrix, shape (33, 33) in complex128
      p_ps: active power pseudo-measurements in pu, shape (B, 33)
      q_ps: reactive power pseudo-measurements in pu, shape (B, 33)
      keep_buses: indices of non-slack buses to evaluate residual on (default 1..32)

    Returns:
      Scalar physics loss tensor in torch.float64.
    """
    if Y.dtype != torch.complex128:
        Y = Y.to(torch.complex128)
    if vm.dtype != torch.float64:
        vm = vm.to(torch.float64)
    if va.dtype != torch.float64:
        va = va.to(torch.float64)

    # Polar to complex phasor: V = |V| * exp(j * theta)
    V = torch.polar(vm, va).to(torch.complex128)  # (B, 33)

    # Complex power injection: S_i = V_i * conj((Y V)_i)
    I_inj = torch.matmul(V, Y.t())                 # (B, 33)
    S_calc = V * torch.conj(I_inj)                 # (B, 33)

    rP = S_calc.real - p_ps.to(torch.float64)
    rQ = S_calc.imag - q_ps.to(torch.float64)

    if keep_buses is not None:
        if not isinstance(keep_buses, torch.Tensor):
            keep_buses = torch.tensor(keep_buses, dtype=torch.long, device=vm.device)
        rP = rP[:, keep_buses]
        rQ = rQ[:, keep_buses]

    return torch.mean(rP ** 2 + rQ ** 2)


def load_ybus(
    meta_path: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load complex bus admittance matrix Y_bus from meta.npz as torch.complex128 tensor.
    """
    d = np.load(meta_path)
    Yr = d["Ybus_real"].astype(np.float64)
    Yi = d["Ybus_imag"].astype(np.float64)
    Y = Yr + 1j * Yi
    tensor = torch.as_tensor(Y, dtype=torch.complex128)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ==============================================================================
# 6. VERIFICATION SMOKE TEST
# ==============================================================================
if __name__ == "__main__":
    print("\n--- Verifying StateEstimatorTCN Architecture ---")
    model = StateEstimatorTCN(
        in_channels=114,
        out_channels=66,
        hidden_channels=[64, 64, 64, 64],
        kernel_size=3,
        dropout=0.1,
        act="tanh",
        use_weight_norm=True,
    )
    print(f"Model created successfully. Receptive field: {model.receptive_field} time steps")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with (B, C_in, T)
    B, C_in, T = 4, 114, 20
    x_test1 = torch.randn(B, C_in, T, dtype=torch.float64)
    y_test1 = model(x_test1)
    print(f"Input shape (B, C_in, T): {tuple(x_test1.shape)} -> Output shape: {tuple(y_test1.shape)}, Dtype: {y_test1.dtype}")
    assert y_test1.shape == (B, 66), f"Expected (B, 66), got {y_test1.shape}"
    assert y_test1.dtype == torch.float64, f"Expected float64, got {y_test1.dtype}"

    # Test forward pass with (B, T, C_in)
    x_test2 = torch.randn(B, T, C_in, dtype=torch.float64)
    y_test2 = model(x_test2)
    print(f"Input shape (B, T, C_in): {tuple(x_test2.shape)} -> Output shape: {tuple(y_test2.shape)}")
    assert y_test2.shape == (B, 66)

    # Test full sequence return
    y_seq = model(x_test1, return_sequences=True)
    print(f"Sequence output shape: {tuple(y_seq.shape)}")
    assert y_seq.shape == (B, T, 66)

    # Test strict causality: future inputs must not affect past outputs
    model.eval()
    with torch.no_grad():
        x_base = torch.randn(1, C_in, 10, dtype=torch.float64)
        x_perturbed = x_base.clone()
        # Perturb only the last time step (t = 9)
        x_perturbed[:, :, -1] += 100.0

        out_base_seq = model(x_base, return_sequences=True)
        out_pert_seq = model(x_perturbed, return_sequences=True)

        # Outputs at t = 0..8 must be IDENTICAL (diff == 0.0)
        past_diff = (out_base_seq[:, :-1, :] - out_pert_seq[:, :-1, :]).abs().max().item()
        future_diff = (out_base_seq[:, -1, :] - out_pert_seq[:, -1, :]).abs().max().item()

        print(f"Causality Check: Max past difference (t < 9): {past_diff:.2e}")
        print(f"Causality Check: Perturbed step difference (t = 9): {future_diff:.2e}")
        assert past_diff < 1e-12, f"Causality violation! Future perturbation leaked into past: diff={past_diff}"
        assert future_diff > 1e-3, "Perturbation had no effect on current step."

    print("All architecture and causality assertions passed!\n")
