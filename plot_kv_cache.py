import re
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Current folder
log_folder = "."
# Output folder for CSV and plots
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)

# Regular expression to extract timestamp, GPU KV cache usage, and Prefix cache hit rate
pattern = re.compile(
    r'INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%'
)

# Iterate over all .log files in current folder
for filename in os.listdir(log_folder):
    if filename.endswith(".log"):
        log_file = os.path.join(log_folder, filename)
        csv_file = os.path.join(output_folder, filename.replace(".log", ".csv"))
        plot_file = os.path.join(output_folder, filename.replace(".log", "_plot.png"))

        timestamps = []
        kv_usage_list = []
        prefix_hit_list = []

        # Read log and extract data
        with open(log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    timestamp_str = match.group(1)
                    timestamp = datetime.strptime(timestamp_str, "%m-%d %H:%M:%S")
                    kv_usage = float(match.group(2))
                    prefix_hit = float(match.group(3))
                    timestamps.append(timestamp)
                    kv_usage_list.append(kv_usage)
                    prefix_hit_list.append(prefix_hit)

        if timestamps:
            # Save CSV
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "GPU_KV_Cache_Usage(%)", "Prefix_Cache_Hit_Rate(%)"])
                for t, kv, ph in zip(timestamps, kv_usage_list, prefix_hit_list):
                    writer.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), kv, ph])

            print(f"{filename}: Extracted {len(timestamps)} entries to {csv_file}")

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, kv_usage_list, label="GPU KV Cache Usage (%)", marker='o')
            plt.plot(timestamps, prefix_hit_list, label="Prefix Cache Hit Rate (%)", marker='x')
            plt.xlabel("Time")
            plt.xticks([])
            plt.ylabel("Percentage (%)")
            plt.ylim(0, 100)
            plt.title(f"GPU KV Cache & Prefix Hit Rate")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            #plt.xticks(rotation=45)

            # Save plot
            plt.savefig(plot_file)
            plt.close()
            print(f"Plot saved as {plot_file}")
