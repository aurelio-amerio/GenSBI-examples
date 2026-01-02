# v1
ae_params = AutoEncoderParams(
        resolution=8192,
        in_channels=2,
        ch=32,
        out_ch=2,
        ch_mult=[
            1,  # 8192
            2,  # 4096
            4,  # 2048
            8,  # 1024
            16,  # 512
            16,  # 256
            16,  # 128
            16,  # 64
            16, # 32
            16, # 16
            16, # 8
            16, # 4
        ],
        num_res_blocks=1,
        z_channels=128,
        scale_factor=0.3611,
        shift_factor=0.1159,
        rngs=nnx.Rngs(42),
        param_dtype=jnp.bfloat16,
    )

v2
params_flux = Flux1Params(
        in_channels=ch_obs,
        vec_in_dim=None,
        context_in_dim=z_ch,
        mlp_ratio=4,
        num_heads=4,
        depth=4,
        depth_single_blocks=8,
        axes_dim=[
            10,
        ],
        dim_obs=dim_obs,
        dim_cond=dim_cond_latent,
        theta = 10*dim_joint,
        qkv_bias=True,
        guidance_embed=False,
        rngs=nnx.Rngs(0),
        param_dtype=jnp.bfloat16,
    )

ae_params = AutoEncoderParams(
        resolution=8192,
        in_channels=2,
        ch=32,
        out_ch=2,
        ch_mult=[
            1,  # 8192
            2,  # 4096
            4,  # 2048
            8,  # 1024
            16,  # 512
            16,  # 256
            16,  # 128
            16,  # 64
            16, # 32
            16, # 16
            16, # 8
            16, # 4
        ],
        num_res_blocks=1,
        z_channels=512,
        scale_factor=0.3611,
        shift_factor=0.1159,
        rngs=nnx.Rngs(42),
        param_dtype=jnp.bfloat16,
    )