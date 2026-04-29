#!/usr/bin/env python3
"""
Latency Correlation Report: Correlates INTERP_MS with token counts in benchmarks log.
Identifies if spikes driven by prompt size or reasoning (completion tokens).
Usage: python latency_correlation_report.py [log_path]
Outputs: console summary, latency_corr.png scatter plot (INTERP vs TOTAL, colored by prompt norm).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

def find_today_log():
    today = datetime.date.today().strftime('%Y-%m-%d')
    return f'benchmarks/bench_{today}.log'

def analyze(log_path):
    try:
        # Cols: 4=INTERP_MS, 6=PROMPT, 7=COMPLETION, 8=TOTAL
        data = np.genfromtxt(log_path, delimiter='\t', skip_header=1, usecols=(4,6,7,8))
        interp, prompt, comp, total = data[:,0], data[:,1], data[:,2], data[:,3]
        corrs = {
            'PROMPT': np.corrcoef(interp, prompt)[0,1],
            'COMPLETION': np.corrcoef(interp, comp)[0,1],
            'TOTAL': np.corrcoef(interp, total)[0,1]
        }
        print('Correlations INTERP_MS vs:')
        for k,v in corrs.items():
            print(f'  {k}: {v:.4f}')
        mean_i = np.mean(interp)
        std_i = np.std(interp)
        thresh = mean_i + 2 * std_i
        spike_idx = np.where(interp > thresh)[0]
        print(f'\nSpikes (> {thresh:.0f}ms, mean {mean_i:.0f}, std {std_i:.0f}, max {np.max(interp):.0f}):')
        for i in spike_idx:
            print(f'  Idx {i}: INTERP={interp[i]:.0f}, PROMPT={prompt[i]:.0f}, TOTAL={total[i]:.0f}, COMP={comp[i]:.0f}')
        if len(spike_idx) == 0:
            print('  No spikes above threshold.')
        # Plot
        norm_prompt = prompt / np.max(prompt)
        plt.figure(figsize=(10,6))
        scatter = plt.scatter(total, interp, c=norm_prompt, cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Prompt Tokens (norm)')
        plt.xlabel('Total Tokens')
        plt.ylabel('INTERP_MS')
        plt.title(f'INTERP_MS vs Total Tokens (colored by Prompt size)\nCorr TOTAL: {corrs["TOTAL"]:.3f}')
        plt.savefig('latency_corr.png', dpi=150, bbox_inches='tight')
        print('\nPlot saved: latency_corr.png')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else find_today_log()
    print(f'Analyzing {path}')
    analyze(path)