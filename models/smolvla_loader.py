import logging
import sys

import numpy as np
import torch

# Add lerobot to path if installed as editable dev install
sys.path.insert(0, "/home/rushabh-jetson/lerobot/src")

# Jetson workaround: transformers 5.x caching_allocator_warmup fails on
# Jetson's unified memory due to NVML internal assertions. Patch to no-op.
try:
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **kw: None
except Exception:
    pass

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from models.preprocessing import preprocess_obs

logger = logging.getLogger(__name__)


class SmolVLAInference:
    """Wrapper around SmolVLAPolicy for closed-loop inference."""

    def __init__(self, model_id: str, device: str, use_fp16: bool, instruction: str):
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.policy = None
        self._tokens = None
        self._dtype = torch.float16 if use_fp16 else torch.float32

    def load(self):
        """Load model weights and pre-tokenize the instruction."""
        logger.info(f"Loading SmolVLA from {self.model_id} ...")

        # Load weights on CPU first to avoid Jetson NVML assertion failures in
        # transformers 5.x CUDACachingAllocator during weight materialization.
        # The pretrained config has device="cuda" hard-coded, so we patch the
        # config dataclass device field after loading it.
        from lerobot.configs.policies import PreTrainedConfig as _LPC
        cfg = _LPC.from_pretrained(self.model_id)
        cfg.device = "cpu"
        # load_vlm_weights=True would call AutoModelForImageTextToText.from_pretrained
        # with device_map="auto", loading the VLM backbone onto CUDA immediately and
        # consuming ~1.5GB before we can control placement. Setting False means the VLM
        # is created as an empty structure; weights come from the safetensors file below.
        cfg.load_vlm_weights = False
        self.policy = SmolVLAPolicy(cfg)

        # Load safetensors weights onto CPU
        from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE
        from huggingface_hub import hf_hub_download
        model_file = hf_hub_download(repo_id=self.model_id, filename=SAFETENSORS_SINGLE_FILE)
        SmolVLAPolicy._load_as_safetensor(self.policy, model_file, "cpu", strict=False)

        logger.info("Weights loaded on CPU. Moving to target device ...")
        # Convert to fp16 on CPU BEFORE moving to CUDA so the GPU allocation
        # is ~1GB (fp16) not ~2GB (fp32). Order matters on memory-constrained Jetson.
        if self.use_fp16:
            self.policy = self.policy.half()
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        logger.info("Model loaded. Tokenizing instruction ...")

        self._tokens = self._tokenize(self.instruction)
        logger.info("SmolVLA ready.")

    def _tokenize(self, instruction: str) -> dict:
        """Tokenize instruction once using the VLM's tokenizer."""
        tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        max_len = self.policy.config.tokenizer_max_length

        # Ensure instruction ends with newline (required by SmolVLM tokenizer)
        if not instruction.endswith("\n"):
            instruction = instruction + "\n"

        encoded = tokenizer(
            [instruction],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def predict(self, obs: dict) -> np.ndarray:
        """Run one inference step.

        Args:
            obs: Dict from SO100Env with keys "top", "wrist", "joint_positions"

        Returns:
            np.ndarray of shape (6,) — joint position targets
        """
        batch = preprocess_obs(obs, self._tokens, torch.device(self.device), self._dtype)

        # autocast ensures float32 tensors (e.g. noise from sample_noise) are
        # automatically cast to fp16 for matrix ops when the model is in fp16.
        device_type = self.device.split(":")[0]
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=self.use_fp16):
            with torch.no_grad():
                action = self.policy.select_action(batch)

        # action shape: (1, 6) — move to CPU and return as numpy
        return action.squeeze(0).cpu().float().numpy()
