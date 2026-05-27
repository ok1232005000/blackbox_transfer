import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- Configuration ---
BASE_DIR = 'd:\\java\\javacode\\weilai'
WHITEBOX_RESULTS_PATH = os.path.join(BASE_DIR, 'experiments', 'whitebox_results.json')
RECORDS_PATH = os.path.join(BASE_DIR, 'experiments', 'records.jsonl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Plotting Functions ---

def plot_whitebox_asr(data):
    """Plots the white-box attack success rate."""
    df = pd.DataFrame.from_dict(data, orient='index')
    df['whitebox_asr_percent'] = df['whitebox_asr'] * 100
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(x=df.index, y='whitebox_asr_percent', data=df, ax=ax, palette='viridis')
    
    ax.set_title('White-box Attack Success Rate (ASR)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Surrogate Model and Epsilon', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, 110)
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 9),
                    textcoords='offset points', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'whitebox_asr.png')
    plt.savefig(save_path)
    print(f"White-box ASR plot saved to {save_path}")
    plt.close()

def plot_blackbox_transfer_asr(df):
    """Plots the black-box transfer attack success rate."""
    # Calculate ASR by grouping
    asr_df = df.groupby(['surrogate', 'eps', 'target_model'])['attack_success'].mean().reset_index()
    asr_df['attack_success_percent'] = asr_df['attack_success'] * 100
    
    # Create a combined column for the x-axis
    asr_df['surrogate_eps'] = asr_df['surrogate'] + '_' + asr_df['eps']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create the plot using catplot for easy faceting by hue
    g = sns.catplot(
        data=asr_df,
        x='surrogate_eps',
        y='attack_success_percent',
        hue='target_model',
        kind='bar',
        height=7,
        aspect=1.8,
        palette='muted',
        legend=False
    )
    
    g.fig.suptitle('Black-box Transfer Attack Success Rate (ASR)', fontsize=16, fontweight='bold', y=1.03)
    g.set_axis_labels('Surrogate Model and Epsilon', 'Success Rate (%)')
    g.set_xticklabels(rotation=45, ha='right')
    g.set(ylim=(0, 110))

    # Add percentage labels
    ax = g.facet_axis(0, 0)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points', fontsize=8)
    
    plt.legend(title='Target Model')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(OUTPUT_DIR, 'blackbox_transfer_asr.png')
    plt.savefig(save_path)
    print(f"Black-box transfer ASR plot saved to {save_path}")
    plt.close()

# --- Main --- 
if __name__ == "__main__":
    # --- White-box Analysis ---
    print("Analyzing white-box results...")
    try:
        with open(WHITEBOX_RESULTS_PATH, 'r') as f:
            whitebox_data = json.load(f)
        plot_whitebox_asr(whitebox_data)
    except FileNotFoundError:
        print(f"Error: White-box results file not found at {WHITEBOX_RESULTS_PATH}")
    except Exception as e:
        print(f"An error occurred during white-box analysis: {e}")

    # --- Black-box Analysis ---
    print("\nAnalyzing black-box results...")
    try:
        records_df = pd.read_json(RECORDS_PATH, lines=True)
        if not records_df.empty:
            plot_blackbox_transfer_asr(records_df)
        else:
            print("Black-box records file is empty. Skipping analysis.")
    except FileNotFoundError:
        print(f"Error: Black-box records file not found at {RECORDS_PATH}")
    except Exception as e:
        print(f"An error occurred during black-box analysis: {e}")
