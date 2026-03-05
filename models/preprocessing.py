import numpy as np
import torch


def preprocess_obs(obs: dict, tokens: dict, device: torch.device, dtype: torch.dtype) -> dict:
    """Convert env obs dict to a SmolVLA-ready batch dict.

    Args:
        obs: Dict with keys "top" (uint8 HxWx3), "wrist" (uint8 HxWx3),
             "joint_positions" (float64, shape (6,))
        tokens: Pre-tokenized instruction dict with "input_ids" and "attention_mask"
        device: Target torch device
        dtype: Target dtype (float32 or float16)

    Returns:
        Batch dict ready for SmolVLAPolicy.select_action()
    """
    # Convert images: uint8 (H,W,C) -> float (1,C,H,W) in [0,1]
    top = _img_to_tensor(obs["top"], device, dtype)
    wrist = _img_to_tensor(obs["wrist"], device, dtype)

    # Convert state: float64 (6,) -> float (1,6)
    state = torch.from_numpy(obs["joint_positions"].astype(np.float32)).to(device=device, dtype=dtype).unsqueeze(0)

    # Map to model-expected keys (pretrained smolvla_base uses camera1/camera2)
    batch = {
        "observation.images.camera1": top,
        "observation.images.camera2": wrist,
        "observation.state": state,
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].to(device=device, dtype=torch.bool),
    }
    return batch


def _img_to_tensor(img: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """uint8 (H,W,C) numpy -> float (1,C,H,W) tensor in [0,1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).to(dtype=dtype) / 255.0
    return t.unsqueeze(0).to(device)
