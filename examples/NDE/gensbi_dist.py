# %%
import numpy as np

import matplotlib.pyplot as plt
from io import BytesIO

# %%


def get_gensbi_samples(n_samples: int = 10000, dpi: int = 100):
    """
    Generates 2D samples from a distribution defined by the text "GenSBI".

    Args:
        n_samples (int): Number of samples to generate.
        dpi (int): Resolution of the mask used for sampling.

    Returns:
        np.ndarray: Array of shape (n_samples, 2) containing the generated samples.
    """
    # Create a figure with a 2:1 aspect ratio
    fig = plt.figure(figsize=(4, 2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Render white text on black background
    ax.text(
        0.5,
        0.5,
        "GenSBI",
        ha="center",
        va="center",
        fontsize=60,
        color="white",
        weight="bold",
    )
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # Draw the canvas and extract the buffer
    fig.canvas.draw()

    # Convert to numpy array
    try:
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        data = data.reshape((h, w, 3))
    except AttributeError:
        # Fallback for newer matplotlib versions or different backends
        # if tostring_rgb is deprecated or unavailable
        try:
            buffer = fig.canvas.buffer_rgba()
            data = np.asarray(buffer)
            # Take only RGB channels if RGBA
            if data.shape[2] == 4:
                data = data[:, :, :3]
        except Exception:
            # Last resort: save to buffer
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor="black")
            buf.seek(0)
            image = plt.imread(buf)
            # If float [0, 1], convert to uint8 [0, 255]
            if image.dtype == np.float32 or image.dtype == np.float64:
                data = (image * 255).astype(np.uint8)
            else:
                data = image
            # Handle RGBA
            if data.shape[2] == 4:
                data = data[:, :, :3]

    plt.close(fig)

    # Convert to grayscale (take one channel since it's black and white) and interpret as probability
    # We drew white text (255) on black background (0).
    # So `mask` is high where we want samples.
    mask = data[:, :, 0].astype(float)

    # Flatten and normalize to get PDF
    flat_mask = mask.flatten()
    prob = flat_mask / flat_mask.sum()

    # Generate random indices based on the probability distribution
    # Using random choice with replacement (as expected for i.i.d samples)
    indices = np.random.choice(len(flat_mask), size=n_samples, p=prob)

    # Convert indices back to (row, col) coordinates
    # Note: row corresponds to y (inverted), col corresponds to x
    rows, cols = np.unravel_index(indices, mask.shape)

    # Add uniform noise for continuous distribution (dithering)
    # Because pixels represent a square area, we add U(-0.5, 0.5)
    rows = rows.astype(float) + np.random.uniform(-0.5, 0.5, size=n_samples)
    cols = cols.astype(float) + np.random.uniform(-0.5, 0.5, size=n_samples)

    # Normalize coordinates to range [-2, 2] for x and [-1, 1] for y (preserving 2:1 ratio)
    # Current range: rows in [0, h], cols in [0, w]
    # In image coordinates, y increases downwards. We usually want y increasing upwards.
    # So map rows -> y: 0 -> max_y, h -> min_y.
    # Actually, let's map directly:
    # x = (col / w) * 4 - 2  (range -2 to 2)
    # y = 1 - (row / h) * 2  (range -1 to 1)
    # Note: w should be roughly 2*h if figsize is (4, 2)

    h, w = mask.shape
    x = (cols / w) * 4.0 - 2.0
    y = 1.0 - (rows / h) * 2.0

    return np.column_stack([x, y])


# # %%
# n_samples = 1_000_000
# samples = get_gensbi_samples(n_samples=n_samples)

# # Plotting
# plt.figure(figsize=(10, 5))
# # plt.hexbin(samples[:, 0], samples[:, 1], gridsize=(100, 50), cmap="Blues")
# plt.hexbin(
#     samples[:, 0],
#     samples[:, 1],
#     gridsize=(200, 100),
#     cmap=transparent_cmap,
#     extent=[-2, 2, -1, 1],
# )
# # plt.hist2d(
# #     samples[:, 0],
# #     samples[:, 1],
# #     bins=(200, 100),
# #     cmap=transparent_cmap,
# #     range=[[-2, 2], [-1, 1]],
# # )
# # plt.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap="Blues")
# plt.xlim(-2, 2)
# plt.ylim(-1, 1)
# #hide axes
# # plt.axis("off")
# plt.grid(None)
# plt.xticks([])
# plt.yticks([])
# # plt.savefig("gensbi_dist.png")
# # print(f"Generated {n_samples} samples and saved plot to gensbi_dist.png")
# plt.show()
# # %%
