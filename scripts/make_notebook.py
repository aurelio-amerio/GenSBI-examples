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
    # target: ../examples/sbi-benchmarks/{task_name_gensbi}/{model_name}/
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

    try:
        with open(stub_path, "r") as f:
            content = f.read()

        # Perform replacements
        for key, value in replacements.items():
            content = content.replace(key, value)

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Write the new file
        with open(target_path, "w") as f:
            f.write(content)

        # call jupytext to convert the python file to a notebook
        # jupytext.convert(target_path, target_path.replace(".py", ".ipynb"))
        notebook = jupytext.read(target_path)
        jupytext.write(notebook, target_path.replace(".py", ".ipynb"))

        print(
            f"Successfully created notebook at: {target_path.replace('.py', '.ipynb')}"
        )

        # remove the python file
        os.remove(target_path)

    except FileNotFoundError:
        print(f"Error: Could not find checking {stub_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    for replacement in replacements:
        make_notebook(replacement)
