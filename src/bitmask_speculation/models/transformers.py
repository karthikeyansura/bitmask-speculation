"""Full-prefix adapter for an already loaded Transformers causal language model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class TransformersGreedyModel:
    """Uncached, single-device reference with raw-logit greedy selection.

    Load the model, move it to its intended device, and call ``eval()`` before
    constructing the adapter. No tokenizer, weights, or device is chosen here.
    Generation-config processors, sampling and model offloading are not applied.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        import torch

        if model.training:
            raise ValueError("call model.eval() before constructing the adapter")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("the reference requires a decoder-only causal model")
        vocab_size = model.config.vocab_size
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("model.config.vocab_size must be an integer")
        if vocab_size <= 0:
            raise ValueError("model.config.vocab_size must be positive")
        devices = {parameter.device for parameter in model.parameters()}
        devices.update(buffer.device for buffer in model.buffers())
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("model tensors must reside on one materialized device")
        self._device = next(iter(devices))
        self._model = model
        self._torch = torch
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def next_token(self, prefix: tuple[int, ...]) -> int:
        """Predict after the full prefix; logits from earlier positions are unused."""
        if not prefix:
            raise ValueError("prefix must contain at least one token")
        if self._model.training:
            raise ValueError("model must remain in evaluation mode")
        torch = self._torch
        with torch.inference_mode():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=self._device)
            output = self._model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = output.logits
            expected = (1, len(prefix), self.vocab_size)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
                raise ValueError(f"expected model logits with shape {expected}")
            if not logits.is_floating_point():
                raise TypeError("model logits must have a floating-point dtype")
            scores = logits[0, -1]
            if torch.isnan(scores).any().item():
                raise ValueError("next-token logits contain NaN")
            return int(scores.argmax().item())
