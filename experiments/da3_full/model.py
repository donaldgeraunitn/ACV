"""Single-view DA3 metric geometry and learned features for SHARP's Gaussian decoder."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sharp.models.composer import GaussianComposer
from sharp.models.gaussian_decoder import create_gaussian_decoder
from sharp.models.heads import DirectPredictionHead
from sharp.models.initializer import create_initializer
from sharp.models.params import PredictorParams
from torch import nn

DA3_LAYERS = (4, 11, 17, 23)
SHARP_CHANNELS = (256, 256, 512, 1024, 1024)
SHARP_STRIDES = (2, 4, 8, 16, 32)
DECODER_TAIL_MODULES = (
    "feature_model.fusion",
    "feature_model.texture_head",
    "feature_model.geometry_head",
    "prediction_head",
)


class FrozenDA3:
    """Extract depth and four transformer feature maps in the same DA3 forward pass."""

    def __init__(self, checkpoint, device, process_res=504):
        """Load the official single-view metric checkpoint on the extraction device."""
        from depth_anything_3.api import DepthAnything3

        self.device = torch.device(device)
        self.model = DepthAnything3.from_pretrained(str(checkpoint))
        if self.model.model_name != "da3metric-large":
            raise ValueError("Expected the metric DA3METRIC-LARGE checkpoint.")
        self.model.eval().requires_grad_(False).to(self.device)
        self.process_res = process_res

    @torch.no_grad()
    def extract(self, image: np.ndarray, fx: float, fy: float) -> dict:
        """Return cacheable CPU features and axial depth in metres for one source image."""
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("DA3 input must be an HWC uint8 RGB image.")
        if not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
            raise ValueError("Expected positive source focal lengths in pixels.")
        pixels, _, _ = self.model.input_processor(
            [image],
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
            num_workers=1,
        )
        output = self.model(pixels[None].to(self.device), export_feat_layers=list(DA3_LAYERS))
        h, w = image.shape[:2]
        depth = output["depth"].reshape(1, 1, *pixels.shape[-2:]).float().cpu()
        ph, pw = depth.shape[-2:]
        # DA3METRIC-LARGE assumes a 300 px focal length at its processing resolution.
        depth = depth * ((fx * pw / w + fy * ph / h) / 600.0)
        if not torch.isfinite(depth).all() or depth.min() <= 0:
            raise ValueError("DA3 returned invalid metric depth.")
        features = []
        for layer in DA3_LAYERS:
            tokens = output["aux"][f"feat_layer_{layer}"][:, 0].permute(0, 3, 1, 2)
            if tuple(tokens.shape) != (1, 1024, ph // 14, pw // 14):
                raise ValueError(f"Unexpected DA3 feature shape at layer {layer}: {tokens.shape}")
            features.append(tokens.to(device="cpu", dtype=torch.float16).contiguous())
        return {"depth": depth.contiguous(), "features": features}


def resize(tensor, size):
    """Resize in SHARP's corner-aligned convention."""
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=True)


class FeatureAdaptor(nn.Module):
    """Project DA3's token grid before upsampling into the five SHARP skip levels."""

    def __init__(self):
        """Create a separate learned projection for each Gaussian decoder input."""
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1024, 24, 1),
                    nn.GroupNorm(4, 24),
                    nn.GELU(),
                    nn.Conv2d(24, channels, 1),
                )
                for channels in SHARP_CHANNELS
            ]
        )

    def forward(self, features, size):
        """Align the projected feature maps with SHARP's spatial pyramid."""
        h, w = size
        # The two finest SHARP skips use separate projections of the earliest DA3 features.
        sources = (features[0], *features)
        return [
            resize(project(source), (h // stride, w // stride))
            for project, source, stride in zip(self.projects, sources, SHARP_STRIDES, strict=True)
        ]


class SecondDepthHead(nn.Module):
    """Predict a signed log-depth correction initialized to zero."""

    def __init__(self):
        """Initialize the second depth map as an exact copy of the first."""
        super().__init__()
        self.features = nn.Conv2d(1024, 16, 1)
        self.correction = nn.Sequential(
            nn.Conv2d(20, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(), nn.Conv2d(24, 1, 1)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, image, depth, early_features):
        """Predict a positive second depth from RGB, DA3 depth, and DA3 features."""
        h, w = image.shape[-2:]
        size = (h // 4, w // 4)
        log_z = resize(depth, size).log()
        context = torch.cat(
            (
                resize(self.features(early_features), size),
                resize(image, size),
                log_z - log_z.mean((-2, -1), keepdim=True),
            ),
            dim=1,
        )
        log_ratio = resize(self.correction(context), (h, w))
        return depth * log_ratio.exp()


class FullDA3Predictor(nn.Module):
    """Construct two DA3 depth channels and refine Gaussians with SHARP's decoder."""

    def __init__(self, sharp_checkpoint: str | Path, *, tune_decoder_tail: bool = False):
        """Load the Gaussian decoder and prediction head, then add the trainable modules."""
        super().__init__()
        p = PredictorParams()
        self.init_model = create_initializer(p.initializer)
        self.feature_model = create_gaussian_decoder(p.gaussian_decoder, list(SHARP_CHANNELS))
        self.prediction_head = DirectPredictionHead(
            self.feature_model.dim_out, self.init_model.num_layers
        )
        self.gaussian_composer = GaussianComposer(
            delta_factor=p.delta_factor,
            min_scale=p.min_scale,
            max_scale=p.max_scale,
            color_activation_type=p.color_activation_type,
            opacity_activation_type=p.opacity_activation_type,
            color_space=p.color_space,
            base_scale_on_predicted_mean=p.base_scale_on_predicted_mean,
            scale_factor=p.gaussian_decoder.stride // p.initializer.stride,
        )
        # mmap loads checkpoint metadata; only these two sets of tensor pages are copied.
        weights = torch.load(sharp_checkpoint, weights_only=True, map_location="cpu", mmap=True)
        for name in ("feature_model", "prediction_head"):
            prefix = name + "."
            group = {
                key.removeprefix(prefix): value
                for key, value in weights.items()
                if key.startswith(prefix)
            }
            getattr(self, name).load_state_dict(group, strict=True)
        self.requires_grad_(False)
        self._save_decoder_tail = tune_decoder_tail
        if tune_decoder_tail:
            for name in DECODER_TAIL_MODULES:
                self.get_submodule(name).requires_grad_(True)
        self.adapter = FeatureAdaptor()
        self.second_head = SecondDepthHead()

    def forward(self, rgb, depth, features):
        """Return both the refined NDC Gaussians and the two metric depth maps."""
        h, w = rgb.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("SHARP's five fusion levels require input dimensions divisible by 32.")
        if len(features) != len(DA3_LAYERS):
            raise ValueError("Expected four DA3 feature levels.")
        depth = resize(depth.to(rgb), (h, w))
        features = [x.to(rgb) for x in features]
        second = self.second_head(rgb, depth, features[0])
        layers = torch.cat((depth, second), dim=1)
        init = self.init_model(rgb, layers)
        encodings = self.adapter(features, (h, w))
        decoded = self.feature_model(init.feature_input, encodings)
        delta = self.prediction_head(decoded)
        gaussians = self.gaussian_composer(delta, init.gaussian_base_values, init.global_scale)
        return gaussians, layers

    def trainable_state_dict(self):
        """Collect the weights of the experiment-specific modules for a compact checkpoint."""
        state = {"adapter": self.adapter.state_dict(), "second_head": self.second_head.state_dict()}
        if self._save_decoder_tail:
            state["decoder_tail"] = {
                name: self.get_submodule(name).state_dict() for name in DECODER_TAIL_MODULES
            }
        return state

    def load_trainable_state_dict(self, state, *, initialize_from_frozen=False):
        """Load an experiment checkpoint, optionally starting tail tuning from official weights."""
        keys = {"adapter", "second_head"}
        if state.keys() - {"decoder_tail"} != keys:
            raise ValueError("Expected weights for the DA3 adapter and learned second depth head.")
        tail = state.get("decoder_tail")
        if tail is None and self._save_decoder_tail and not initialize_from_frozen:
            raise ValueError("Missing decoder-tail weights in checkpoint.")
        if tail is not None:
            if tail.keys() != set(DECODER_TAIL_MODULES):
                raise ValueError("Checkpoint does not contain the exact Gaussian decoder tail.")
            for name in DECODER_TAIL_MODULES:
                self.get_submodule(name).load_state_dict(tail[name], strict=True)
            self._save_decoder_tail = True
        for name in keys:
            getattr(self, name).load_state_dict(state[name], strict=True)
