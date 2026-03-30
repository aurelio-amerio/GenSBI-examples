import re
import csv
import os

files = [
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/sbibm/condor_logs/outfile_two_moons_diffusion_flux_100000.out",
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/sbibm/condor_logs/outfile_two_moons_diffusion_flux1joint_100000.out",
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/sbibm/condor_logs/outfile_two_moons_flow_flux_100000.out",
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/sbibm/condor_logs/outfile_two_moons_flow_flux1joint_100000.out",
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/sbibm/condor_logs/outfile_two_moons_score_matching_flux_100000.out",
    "/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/sbibm/condor_logs/outfile_two_moons_score_matching_flux1joint_100000.out"
]

results = []

for filepath in files:
    filename = os.path.basename(filepath)
    model_name = filename.replace("outfile_two_moons_", "").replace("_100000.out", "")
    
    with open(filepath, "r") as f:
        content = f.read()
    
    # Extract training time
    train_time_match = re.search(r"Training time:\s+([\d\.]+)\s+seconds", content)
    train_time = float(train_time_match.group(1)) if train_time_match else None
    
    # Extract sampling times
    sampling_times = [float(x) for x in re.findall(r"Sampling time for observation=\d+:\s+([\d\.]+)\s+seconds", content)]
    
    if sampling_times:
        avg_sampling_time = sum(sampling_times) / len(sampling_times)
    else:
        avg_sampling_time = None
        
    training_steps = 10000
    it_per_s = training_steps / train_time if train_time else None
    
    results.append({
        "Task Name": "two moons",
        "Model Name": model_name,
        "Training Time (s)": round(train_time, 2) if train_time else None,
        "Training (it/s)": round(it_per_s, 2) if it_per_s else None,
        "Average Sampling Time per 10000 samples (s)": round(avg_sampling_time, 2) if avg_sampling_time else None
    })

csv_file = "/lhome/ific/a/aamerio/data/github/GenSBI-examples/two_moons_times.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Data successfully saved to {csv_file}")
