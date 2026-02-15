import os
import jupytext

# Dictionary of replacements based on the task requirements

# Bernoulli GLM
replacements_flow_flux = {
    "{model_name}": "flow_flux",
    "{model_architecture}": "Flux1",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "conditional",
}

replacements_flow_flux1joint = {
    "{model_name}": "flow_flux1joint",
    "{model_architecture}": "Flux1Joint",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "joint",
}

replacements_flow_simformer = {
    "{model_name}": "flow_simformer",
    "{model_architecture}": "Simformer",
    "{technique}": "Flow",
    "{matching_technique}": "Flow Matching",
    "{kind}": "conditional",
}

replacements_diffusion_flux = {
    "{model_name}": "diffusion_flux",
    "{model_architecture}": "Flux1",
    "{technique}": "Diffusion",
    "{matching_technique}": "Score Matching",
    "{kind}": "conditional",
}

replacements_diffusion_flux1joint = {
    "{model_name}": "diffusion_flux1joint",
    "{model_architecture}": "Flux1Joint",
    "{technique}": "Diffusion",
    "{matching_technique}": "Score Matching",
    "{kind}": "joint",
}

replacements_diffusion_simformer = {
    "{model_name}": "diffusion_simformer",
    "{model_architecture}": "Simformer",
    "{technique}": "Diffusion",
    "{matching_technique}": "Score Matching",
    "{kind}": "conditional",
}


replacements = [
    replacements_flow_flux,
    replacements_flow_flux1joint,
    replacements_flow_simformer,
    replacements_diffusion_flux,
    replacements_diffusion_flux1joint,
    replacements_diffusion_simformer,
]


def make_notebook(replacements):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the template file
    template_path = os.path.join(script_dir, "two_moons_template.md")

    # Construct the target directory path
    target_dir = os.path.join(
        script_dir,
        "..",
        "examples",
        "sbi-benchmarks",
        "two_moons",
        replacements["{model_name}"],
    )

    # Resolve to absolute path
    target_dir = os.path.abspath(target_dir)

    # Construct the target file name: {task_name_gensbi}_{model_name}.md
    target_filename = f"two_moons_{replacements['{model_name}']}.md"
    target_path = os.path.join(target_dir, target_filename)

    print(f"Reading template from: {template_path}")

    try:
        with open(template_path, "r") as f:
            content = f.read()

        # Perform replacements
        for key, value in replacements.items():
            content = content.replace(key, value)

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        # Write the temporary md file
        with open(target_path, "w") as f:
            f.write(content)

        # Read the md file as a notebook object
        notebook = jupytext.read(target_path, fmt="md:myst")

        # 1. Write as standard Jupyter Notebook (.ipynb)
        ipynb_path = target_path.replace(".md", ".ipynb")
        jupytext.write(notebook, ipynb_path)
        print(f"Created notebook: {ipynb_path}")

        # delete the temporary md file
        os.remove(target_path)

    except FileNotFoundError:
        print(f"Error: Could not find checking {template_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    for replacement in replacements:
        make_notebook(replacement)
