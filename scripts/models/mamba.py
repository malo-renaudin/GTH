from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def build_mamba(cfg, device, dtype):
    config = MambaConfig(
        d_model=cfg["d_model"],
        n_layer=cfg["n_layers"],
        vocab_size=cfg["vocab_size"],
        ssm_cfg={"layer": "Mamba2"},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=16,
    )
    #check in implementation that wieghts are tied between embedding and lm_head
    return MambaLMHeadModel(config, device=device, dtype=dtype)