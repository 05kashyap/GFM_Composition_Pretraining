"""
Exponential Moving Average (EMA) model wrapper for target networks.

Used for BYOL-style self-supervised learning where the target branch uses
EMA-smoothed weights from the online branch.

Example:
    >>> online_decoder = QuerySlotDecoder(...)
    >>> ema = EMAModel(online_decoder, tau=0.996)
    >>>
    >>> # Training loop
    >>> for batch in dataloader:
    >>>     # Forward pass with online (gradients flow)
    >>>     online_slots = online_decoder(x, backbone_feat=feat)
    >>>
    >>>     # Forward pass with EMA target (no gradients)
    >>>     with torch.no_grad():
    >>>         target_slots = ema.module(x, backbone_feat=feat.detach())
    >>>
    >>>     loss = byol_loss(online_slots, target_slots)
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>
    >>>     # Update EMA weights AFTER optimizer step
    >>>     ema.update(online_decoder)
"""

from __future__ import annotations

import copy
from typing import Iterator, Tuple

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average wrapper for neural network modules.

    Maintains an EMA copy of a module's parameters and buffers. The EMA copy
    is automatically initialized from the online module at construction time.

    The EMA update rule is:
        theta_ema = tau * theta_ema + (1 - tau) * theta_online

    where tau is typically 0.996 (BYOL) or 0.999 (MoCo).

    Note on device handling:
        The EMA module may be created before the online module is moved to GPU
        (e.g., in model __init__ before DDP wrapping). The `sync_device()` method
        ensures the EMA module is on the same device as the online module.
        This is automatically called in `update()`, so the EMA module will be
        moved to GPU on the first training iteration.

    Args:
        online_module: The online module to track with EMA.
        tau: EMA decay rate. Higher = slower update. Typical: 0.996 (BYOL).

    Attributes:
        tau: The EMA decay rate.
        _ema_module: The internal EMA copy of the module.
    """

    def __init__(self, online_module: nn.Module, tau: float = 0.996):
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")

        self.tau = tau
        self._device_synced = False

        # Deep copy the online module to create the EMA target
        self._ema_module = copy.deepcopy(online_module)

        # Try to move to same device as online module (may still be CPU at init)
        try:
            device = next(online_module.parameters()).device
            self._ema_module.to(device)
            self._device_synced = True
        except StopIteration:
            # Module has no parameters (unlikely but handle gracefully)
            pass

        # Set to eval mode permanently (no dropout, batchnorm in eval mode)
        self._ema_module.eval()

        # Disable gradients on all EMA parameters
        for param in self._ema_module.parameters():
            param.requires_grad_(False)

    def sync_device(self, online_module: nn.Module) -> None:
        """Ensure EMA module is on the same device as online module.

        This handles the case where the model is moved to GPU after construction
        (e.g., by DDP wrapping). Should be called before using the EMA module
        for forward passes.

        Args:
            online_module: The online module to sync device with.
        """
        try:
            online_device = next(online_module.parameters()).device
            ema_device = next(self._ema_module.parameters()).device

            if online_device != ema_device:
                self._ema_module.to(online_device)
                self._device_synced = True
        except StopIteration:
            pass

    @property
    def module(self) -> nn.Module:
        """Returns the EMA copy of the module.

        Returns:
            The EMA module (in eval mode, no gradients).
        """
        return self._ema_module

    @torch.no_grad()
    def update(self, online_module: nn.Module) -> None:
        """Update EMA weights from the online module.

        Applies the EMA update rule to both parameters and buffers:
            theta_ema = tau * theta_ema + (1 - tau) * theta_online

        This should be called AFTER the optimizer step, not before.

        Args:
            online_module: The online module whose weights to track.
        """
        # Ensure devices are synced first (handles DDP moving model after init)
        self.sync_device(online_module)

        # Update parameters
        for ema_param, online_param in zip(
            self._ema_module.parameters(),
            online_module.parameters()
        ):
            # EMA update: theta_ema = tau * theta_ema + (1 - tau) * theta_online
            ema_param.data.mul_(self.tau).add_(
                online_param.data, alpha=1.0 - self.tau
            )

        # Update buffers (e.g., BatchNorm running stats)
        # Copy directly (not EMA smoothed) - handles batch norm running mean/var
        for ema_buf, online_buf in zip(
            self._ema_module.buffers(),
            online_module.buffers()
        ):
            ema_buf.data.copy_(online_buf.data)

    def state_dict(self) -> dict:
        """Returns the EMA module's state dict.

        Useful for checkpointing if you want to save EMA weights.
        """
        return self._ema_module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load a state dict into the EMA module.

        Args:
            state_dict: State dict to load.
        """
        self._ema_module.load_state_dict(state_dict)

    def named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """Iterate over EMA module's named parameters.

        Useful for filtering these out of optimizer param groups.
        """
        return self._ema_module.named_parameters()

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over EMA module's parameters."""
        return self._ema_module.parameters()


# --------------------------------------------------------------------------- #
# EMA Hook for MMEngine Runner
# --------------------------------------------------------------------------- #

try:
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS

    @HOOKS.register_module()
    class EMAUpdateHook(Hook):
        """Hook to update EMA target network after each training iteration.

        This hook calls ``model.update_ema()`` after the optimizer step in each
        training iteration. It expects the model to have an ``update_ema()`` method
        (e.g., CompositionAwareDynamicVis).

        The hook also provides optional debug logging to verify EMA is working:
        at iteration 10, it prints the norm of the weight difference between
        online and EMA networks.

        Args:
            debug_step: Iteration at which to print debug info. Set to 0 to disable.

        Usage in config::

            custom_hooks = [
                dict(type='EMAUpdateHook', debug_step=10),
            ]
        """

        priority = 'LOW'  # Run after optimizer step

        def __init__(self, debug_step: int = 10):
            self.debug_step = debug_step

        def after_train_iter(
            self,
            runner,
            batch_idx: int,
            data_batch=None,
            outputs=None,
        ) -> None:
            """Update EMA after each training iteration.

            Args:
                runner: The MMEngine runner.
                batch_idx: Index of the current batch within the epoch.
                data_batch: The data batch (unused).
                outputs: The outputs from the model (unused).
            """
            # Get the raw model (unwrap DDP if needed)
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module

            # Call update_ema if the method exists
            if hasattr(model, 'update_ema'):
                model.update_ema()

            # Debug: verify EMA weights differ from online at specified step
            current_iter = runner.iter
            if self.debug_step > 0 and current_iter == self.debug_step:
                self._print_ema_debug(model, current_iter)

        def _print_ema_debug(self, model, current_iter: int) -> None:
            """Print debug info comparing online vs EMA weights."""
            if not hasattr(model, 'slot_decoder') or model.slot_decoder is None:
                return
            if not hasattr(model, 'slot_decoder_ema') or model.slot_decoder_ema is None:
                runner_logger = None
                print(f"[EMA Debug @ iter {current_iter}] EMA not enabled (slot_decoder_ema is None)")
                return

            try:
                online_params = list(model.slot_decoder.parameters())
                ema_params = list(model.slot_decoder_ema.module.parameters())

                if len(online_params) == 0:
                    print(f"[EMA Debug @ iter {current_iter}] No parameters in slot_decoder")
                    return

                # Compute weight difference norm for first parameter
                online_w = online_params[0].data
                ema_w = ema_params[0].data
                diff_norm = (online_w - ema_w).norm().item()

                # Also compute total diff across all params
                total_diff = 0.0
                for op, ep in zip(online_params, ema_params):
                    total_diff += (op.data - ep.data).norm().item() ** 2
                total_diff = total_diff ** 0.5

                print(f"[EMA Debug @ iter {current_iter}] "
                      f"First param diff: {diff_norm:.6f}, "
                      f"Total diff: {total_diff:.6f} "
                      f"(should be ~0 at iter 0, growing slowly)")
            except Exception as e:
                print(f"[EMA Debug @ iter {current_iter}] Error: {e}")

except ImportError:
    # MMEngine not available - skip hook registration
    pass
