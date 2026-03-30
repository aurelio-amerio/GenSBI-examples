import pandas as pd
import sys

def format_duration(secs):
    """Format seconds as HH:MM:SS."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def generate_latex_tables(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # Filter for the 100k samples/steps case
    df = df[df['budget'] == 100000]
    
    if df.empty:
        print("No data found for budget == 100000")
        return

    tasks = df['task'].dropna().unique()
    
    # Methodologies and models
    methods = ["EDM", "FM", "SM"]
    models = ["Flux1", "Flux1Joint"]
    
    out_file = "training_time_tables.tex"
    with open(out_file, "w") as f:
        for task in sorted(tasks):
            task_df = df[df['task'] == task]
            
            print(f"% ==========================================", file=f)
            print(f"% Table for Task: {task}", file=f)
            print(f"% ==========================================", file=f)
            print("\\begin{table}[htbp]", file=f)
            print("  \\centering", file=f)
            print("  \\begin{tabular}{l c c}", file=f)
            print("    \\toprule", file=f)
            print("    Model & it/s & Total Time (HH:MM:SS) \\\\", file=f)
            print("    \\midrule", file=f)
            
            for method in methods:
                for model in models:
                    row_name = f"{method} {model}"
                    
                    # Filter for specific method and model
                    row = task_df[(task_df['method'] == method) & (task_df['model'] == model)]
                    
                    if len(row) > 0:
                        row = row.iloc[0]
                        its = row['its']
                        
                        # Project total time to 100k steps
                        projected_secs = 100000 / its if its > 0 else 0
                        time_str = format_duration(projected_secs)
                        
                        print(f"    {row_name} & {its:.3f} & {time_str} \\\\", file=f)
                    else:
                        print(f"    {row_name} & - & - \\\\", file=f)
                        
            print("    \\bottomrule", file=f)
            print("  \\end{tabular}", file=f)
            
            # Format task name for display
            display_task = task.replace("_", "\\_")
            print(f"  \\caption{{Training performance for 100,000 steps on {display_task}.}}", file=f)
            print(f"  \\label{{tab:training_times_{task}}}", file=f)
            print("\\end{table}", file=f)
            print("\n", file=f)
            
    print(f"Successfully wrote tables to {out_file}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "training_times.csv"
    generate_latex_tables(csv_file)
