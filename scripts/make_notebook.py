import os
import jupytext

# Dictionary of replacements based on the task requirements

# Bernoulli GLM
replacements_bernoulli_glm_flow_flux = {
    "{TASKNAME}": "Bernoulli GLM",
    "{task_name_gensbi}": "bernoulli_glm",
    "{model_name}": "flow_flux",
    "{model_architecture}": "Flux1",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "conditional",
}
replacements_bernoulli_glm_flow_flux1joint = {
    "{TASKNAME}": "Bernoulli GLM",
    "{task_name_gensbi}": "bernoulli_glm",
    "{model_name}": "flow_flux1joint",
    "{model_architecture}": "Flux1Joint",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "joint",
}

# Gaussian Linear
replacements_gaussian_linear_flow_flux = {
    "{TASKNAME}": "Gaussian Linear",
    "{task_name_gensbi}": "gaussian_linear",
    "{model_name}": "flow_flux",
    "{model_architecture}": "Flux1",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "conditional",
}
replacements_gaussian_linear_flow_flux1joint = {
    "{TASKNAME}": "Gaussian Linear",
    "{task_name_gensbi}": "gaussian_linear",
    "{model_name}": "flow_flux1joint",
    "{model_architecture}": "Flux1Joint",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "joint",
}

# Gaussian Mixture
replacements_gaussian_mixture_flow_flux = {
    "{TASKNAME}": "Gaussian Mixture",
    "{task_name_gensbi}": "gaussian_mixture",
    "{model_name}": "flow_flux",
    "{model_architecture}": "Flux1",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "conditional",
}
replacements_gaussian_mixture_flow_flux1joint = {
    "{TASKNAME}": "Gaussian Mixture",
    "{task_name_gensbi}": "gaussian_mixture",
    "{model_name}": "flow_flux1joint",
    "{model_architecture}": "Flux1Joint",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "joint",
}

# SLCP
replacements_slcp_flow_flux = {
    "{TASKNAME}": "SLCP",
    "{task_name_gensbi}": "slcp",
    "{model_name}": "flow_flux",
    "{model_architecture}": "Flux1",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "conditional",
}
replacements_slcp_flow_flux1joint = {
    "{TASKNAME}": "SLCP",
    "{task_name_gensbi}": "slcp",
    "{model_name}": "flow_flux1joint",
    "{model_architecture}": "Flux1Joint",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "joint",
}
replacements_slcp_flow_simformer = {
    "{TASKNAME}": "SLCP",
    "{task_name_gensbi}": "slcp",
    "{model_name}": "flow_simformer",
    "{model_architecture}": "Simformer",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "joint",
}

replacements = [
    replacements_bernoulli_glm_flow_flux,
    replacements_bernoulli_glm_flow_flux1joint,
    replacements_gaussian_linear_flow_flux,
    replacements_gaussian_linear_flow_flux1joint,
    replacements_gaussian_mixture_flow_flux,
    replacements_gaussian_mixture_flow_flux1joint,
    replacements_slcp_flow_flux,
    replacements_slcp_flow_flux1joint,
    replacements_slcp_flow_simformer,
]


def make_notebook(replacements):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the stub file
    stub_path = os.path.join(script_dir, "notebook_stub.txt")

    # Construct the target directory path
    target_dir = os.path.join(
        script_dir,
        "..",
        "examples",
        "sbi-benchmarks",
        replacements["{task_name_gensbi}"],
        replacements["{model_name}"],
    )

    # Resolve to absolute path
    target_dir = os.path.abspath(target_dir)

    # Construct the target file name: {task_name_gensbi}_{model_name}.py
    target_filename = (
        f"{replacements['{task_name_gensbi}']}_{replacements['{model_name}']}.py"
    )
    target_path = os.path.join(target_dir, target_filename)

    print(f"Reading stub from: {stub_path}")

    # read C2ST accuracy and std from file c2st_results_ema_1_flow_flux.txt
    # (Wrapped in try/except to prevent crash if results file is missing)
    try:
        c2st_results_path = os.path.join(
            target_dir,
            "c2st_results",
            f"c2st_results_ema_1_{replacements['{model_name}']}.txt",
        )
        with open(c2st_results_path, "r") as f:
            c2st_results = f.read()

        c2st_accuracy = c2st_results.split("Average C2ST accuracy EMA: ")[1].split(" ")[
            0
        ]
        c2st_std = c2st_results.split("Average C2ST accuracy EMA: ")[1].split(" ")[2]
    except (FileNotFoundError, IndexError):
        print(
            f"Warning: C2ST results not found for {target_filename}, using placeholders."
        )
        c2st_accuracy = "N/A"
        c2st_std = "N/A"

    try:
        with open(stub_path, "r") as f:
            content = f.read()

        # Perform replacements
        for key, value in replacements.items():
            content = content.replace(key, value)
        content = content.replace("{C2ST_ACCURACY}", c2st_accuracy)
        content = content.replace("{C2ST_STD}", c2st_std)

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Write the temporary python file
        with open(target_path, "w") as f:
            f.write(content)

        # Read the python file as a notebook object
        notebook = jupytext.read(target_path)

        # 1. Write as standard Jupyter Notebook (.ipynb)
        ipynb_path = target_path.replace(".py", ".ipynb")
        jupytext.write(notebook, ipynb_path)
        print(f"Created notebook: {ipynb_path}")

        # 2. Write as MyST Markdown (.md)
        myst_path = target_path.replace(".py", ".md")
        jupytext.write(notebook, myst_path, fmt="md:myst")
        print(f"Created MyST Markdown: {myst_path}")

        # Remove the temporary python file
        os.remove(target_path)

    except FileNotFoundError:
        print(f"Error: Could not find checking {stub_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    for replacement in replacements:
        make_notebook(replacement)
