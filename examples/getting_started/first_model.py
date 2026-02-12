# %%
import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "cuda"  # change to 'cpu' if no GPU is available


import grain
import numpy as np
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
from flax import nnx

from gensbi.recipes import Flux1FlowPipeline

from gensbi.utils.plotting import plot_marginals
import matplotlib.pyplot as plt

from gensbi.diagnostics import check_tarp, run_tarp, plot_tarp
from gensbi.diagnostics import check_sbc, run_sbc, sbc_rank_plot
from gensbi.diagnostics import LC2ST, plot_lc2st


def _simulator(key, thetas):

    xs = thetas + 1 + jax.random.normal(key, thetas.shape) * 0.1

    thetas = thetas[..., None]
    xs = xs[..., None]

    # when making a dataset for the joint pipeline, thetas need to come first
    data = jnp.concatenate([thetas, xs], axis=1)

    return data


theta_prior = dist.Uniform(
    low=jnp.array([-2.0, -2.0, -2.0]), high=jnp.array([2.0, 2.0, 2.0])
)


def simulator(key, nsamples):
    theta_key, sample_key = jax.random.split(key, 2)
    thetas = theta_prior.sample(theta_key, (nsamples,))

    return _simulator(sample_key, thetas)


def main():
    dim_obs = (
        3  # dimension of the observation (theta), that is the simulator input shape
    )
    dim_cond = 3  # dimension of the condition (xs), that is the simulator output shape
    dim_joint = dim_obs + dim_cond  # dimension of the joint (theta, xs), useful later

    train_data = jnp.asarray(
        simulator(jax.random.PRNGKey(0), 100_000), dtype=jnp.bfloat16
    )
    val_data = jnp.asarray(simulator(jax.random.PRNGKey(1), 2000), dtype=jnp.bfloat16)

    def split_obs_cond(data):
        return (
            data[:, :dim_obs],
            data[:, dim_obs:],
        )  # assuming first dim_obs are obs, last dim_cond are cond

    batch_size = 256

    train_dataset_grain = (
        grain.MapDataset.source(np.array(train_data))
        .shuffle(42)
        .repeat()
        .to_iter_dataset()
    )
    performance_config = grain.experimental.pick_performance_config(
        ds=train_dataset_grain,
        ram_budget_mb=1024 * 4,
        max_workers=None,
        max_buffer_size=None,
    )
    train_dataset_grain = (
        train_dataset_grain.batch(batch_size)
        .map(split_obs_cond)
        .mp_prefetch(performance_config.multiprocessing_options)
    )

    val_dataset_grain = (
        grain.MapDataset.source(np.array(val_data))
        .shuffle(42)
        .repeat()
        .to_iter_dataset()
        .batch(batch_size)
        .map(split_obs_cond)
    )

    cwd = os.getcwd()
    checkpoint_dir = os.path.join(cwd, "checkpoints")
    config_path = os.path.join(cwd, "config_flow_flux.yaml")

    pipeline = Flux1FlowPipeline.init_pipeline_from_config(
        train_dataset_grain,
        val_dataset_grain,
        dim_obs,
        dim_cond,
        config_path,
        checkpoint_dir,
    )

    loss_history = pipeline.train(nnx.Rngs(0), save_model=True)

    steps = np.linspace(1, len(loss_history[0]), len(loss_history[0])) * 100
    plt.plot(steps, loss_history[0], label="train loss")
    plt.plot(steps, loss_history[1], label="val loss")
    plt.yscale("log")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.ylim(0.1, 10)
    plt.legend()
    plt.savefig(
        "flux1_flow_pipeline_loss_2.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()

    rngs = nnx.Rngs(42)

    new_sample = simulator(
        jax.random.PRNGKey(20), 1
    )  # the observation for which we want to reconstruct the posterior

    true_theta = new_sample[
        :, :dim_obs, :
    ]  # The input used for the simulation, AKA the true value
    x_o = new_sample[
        :, dim_obs:, :
    ]  # The observation from the simulation for which we want to reconstruct the posterior

    samples = pipeline.sample(rngs.sample(), x_o, nsamples=100_000)

    plot_marginals(
        np.array(samples[..., 0]),
        gridsize=30,
        true_param=np.array(true_theta[0, :, 0]),
        range=[(1, 3), (1, 3), (-0.6, 0.5)],
    )
    plt.savefig(
        "flux1_flow_pipeline_marginals_2.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()

    key = jax.random.PRNGKey(1234)

    # sample the dataset
    test_data = simulator(jax.random.PRNGKey(1), 200)

    # split in thetas and xs
    thetas_ = test_data[:, :dim_obs, :]  # (200, 3, 1)
    xs_ = test_data[:, dim_obs:, :]  # (200, 3, 1)

    posterior_samples_ = pipeline.sample_batched(
        jax.random.PRNGKey(0), xs_, nsamples=1000
    )

    # flatten the dataset. sbi expects 2D arrays of shape (num_samples, features), while our data is 3D of shape (num_samples, dim, channels).
    # we reshape a sample of size (dim, channels) into a vector of size (dim * channels)
    thetas = thetas_.reshape(thetas_.shape[0], -1)
    xs = xs_.reshape(xs_.shape[0], -1)
    posterior_samples = posterior_samples_.reshape(
        posterior_samples_.shape[0], posterior_samples_.shape[1], -1
    )

    ranks, dap_samples = run_sbc(thetas, xs, posterior_samples)
    check_stats = check_sbc(ranks, thetas, dap_samples, 1_000)

    # %%
    print(check_stats)

    # %%
    f, ax = sbc_rank_plot(ranks, 1_000, plot_type="hist", num_bins=20)
    plt.savefig(
        "flux1_flow_pipeline_sbc_2.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()

    # %%
    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
    )

    # %%
    atc, ks_pval = check_tarp(ecp, alpha)
    print(atc, "Should be close to 0")
    print(ks_pval, "Should be larger than 0.05")

    # %%
    plot_tarp(ecp, alpha)
    plt.savefig(
        "flux1_flow_pipeline_tarp_2.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()

    key = jax.random.PRNGKey(1234)
    # sample the dataset
    test_data = simulator(jax.random.PRNGKey(1), 10_000)

    # split in thetas and xs
    thetas_ = test_data[:, :dim_obs, :]  # (10_000, 3, 1)
    xs_ = test_data[:, dim_obs:, :]  # (10_000, 3, 1)

    # Generate one posterior sample for every prior predictive.
    posterior_samples_ = pipeline.sample(key, x_o=xs_, nsamples=xs_.shape[0])

    # flatten the dataset. sbi expects 2D arrays of shape (num_samples, features), while our data is 3D of shape (num_samples, dim, channels).
    # we reshape a sample of size (dim, channels) into a vector of size (dim * channels)
    thetas = thetas_.reshape(thetas_.shape[0], -1)
    xs = xs_.reshape(xs_.shape[0], -1)
    posterior_samples = posterior_samples_.reshape(posterior_samples_.shape[0], -1)

    # %%

    # Train the L-C2ST classifier.
    lc2st = LC2ST(
        thetas=thetas,
        xs=xs,
        posterior_samples=posterior_samples,
        classifier="mlp",
        num_ensemble=1,
    )
    _ = lc2st.train_under_null_hypothesis()
    _ = lc2st.train_on_observed_data()

    # %%
    key = jax.random.PRNGKey(12345)

    sample = simulator(key, 1)
    # theta_true_ = sample[:, :dim_obs, :]
    x_o_ = sample[:, dim_obs:, :]

    # Note: x_o must have a batch-dimension. I.e. `x_o.shape == (1, observation_shape)`.
    post_samples_star_ = pipeline.sample(rngs.sample(), x_o_, nsamples=10_000)

    # theta_true = theta_true_.reshape(-1)  # (3,)
    x_o = x_o_.reshape(1, -1)  # (3,)
    post_samples_star = np.array(
        post_samples_star_.reshape(post_samples_star_.shape[0], -1)
    )

    # %%
    probs_data, scores_data = lc2st.get_scores(
        theta_o=post_samples_star,
        x_o=x_o,
        return_probs=True,
        trained_clfs=lc2st.trained_clfs,
    )
    probs_null, scores_null = lc2st.get_statistics_under_null_hypothesis(
        theta_o=post_samples_star,
        x_o=x_o,
        return_probs=True,
    )

    # %%
    conf_alpha = 0.05
    p_value = lc2st.p_value(post_samples_star, x_o)
    reject = lc2st.reject_test(post_samples_star, x_o, alpha=conf_alpha)

    fig, ax = plot_lc2st(
        lc2st,
        post_samples_star,
        x_o,
    )
    plt.savefig(
        "flux1_flow_pipeline_lc2st_2.png", dpi=100, bbox_inches="tight"
    )  # uncomment to save the figure
    plt.show()


if __name__ == "__main__":
    main()
