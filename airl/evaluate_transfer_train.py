import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to your progress file
log_path = "/home/shiv1901/safeil-data-collection-main/safeil-data-collection-main/airl/data/2026-02-26_airl_transfer_v4/2026-02-26_11-13-47-airl_transfer_v4_s0/progress.txt"

if os.path.exists(log_path):
    # safe_rl logs are usually tab-separated
    df = pd.read_table(log_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Learned Reward
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Learned Reward (AIRL)', color=color)
    ax1.plot(df['Epoch'], df['AverageEpRet'], color=color, linewidth=2, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Safety Cost on the same graph (different axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Episode Cost (Safety)', color=color)
    ax2.plot(df['Epoch'], df['AverageEpCost'], color=color, linewidth=2, label='Cost')
    
    # Draw the Cost Limit line
    ax2.axhline(y=25.0, color='black', linestyle='--', alpha=0.7, label='Cost Limit (25)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Safe AIRL Transfer Performance\n(WSL Headless Run)')
    fig.tight_layout()
    
    # Save the file
    plt.savefig('transfer_results_plot.png')
    print("--- SUCCESS ---")
    print("Plot saved as: transfer_results_plot.png")
    
    # Print the final values for quick verification
    last_row = df.iloc[-1]
    print(f"Final Epoch: {int(last_row['Epoch'])}")
    print(f"Final Reward: {last_row['AverageEpRet']:.2f}")
    print(f"Final Cost: {last_row['AverageEpCost']:.2f}")
else:
    print(f"Error: Log file not found at {log_path}")