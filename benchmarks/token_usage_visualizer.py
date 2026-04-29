import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Auto-detect last 7 days' logs
end_date = datetime.now().date()
dates = pd.date_range(end_date - timedelta(days=6), end_date).strftime('%Y-%m-%d')
files = [f"benchmarks/bench_{date}.log" for date in dates if os.path.exists(f"benchmarks/bench_{date}.log")]

print(f"Found {len(files)} log files: {{files}}")

if not files:
    print("No benchmark logs found for last 7 days.")
    exit(1)

dfs = []
for f in files:
    date_str = f.split('_')[-1].replace('.log', '')
    df = pd.read_csv(f, sep=r'\s+', engine='python', header=0)
    df['date'] = date_str
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['PROMPT'] = pd.to_numeric(all_data['PROMPT'], errors='coerce')
all_data['COMPLETION'] = pd.to_numeric(all_data['COMPLETION'], errors='coerce')

# Daily stats
daily_stats = all_data.groupby('date').agg({
    'PROMPT': ['mean', 'sum'],
    'COMPL ETION': ['mean', 'sum']
}).round(0)
daily_stats.columns = ['prompt_mean', 'prompt_total', 'completion_mean', 'completion_total']

print("Daily stats:")
print(daily_stats)

# Plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Scatter: avg prompt vs completion per day
ax1.scatter(daily_stats['prompt_mean'], daily_stats['completion_mean'])
for i, date in enumerate(daily_stats.index):
    ax1.annotate(date, (daily_stats['prompt_mean'].iloc[i], daily_stats['completion_mean'].iloc[i]))
z = np.polyfit(daily_stats['prompt_mean'], daily_stats['completion_mean'], 1)
p = np.poly1d(z)
ax1.plot(daily_stats['prompt_mean'], p(daily_stats['prompt_mean']), "r--")
ax1.set_xlabel('Daily Avg Prompt Tokens')
ax1.set_ylabel('Daily Avg Completion Tokens')
ax1.set_title('Prompt vs Completion Averages (Trend Line)')

# 2. Line plot: daily means
ax2.plot(daily_stats.index, daily_stats['prompt_mean'], label='Prompt', marker='o')
ax2.plot(daily_stats.index, daily_stats['completion_mean'], label='Completion', marker='s')
ax2.set_xlabel('Date')
ax2.set_ylabel('Average Tokens')
ax2.set_title('Daily Average Token Usage')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# 3. Stacked bar: daily totals
width = 0.35
x = range(len(daily_stats))
ax3.bar(x, daily_stats['prompt_total'], width, label='Prompt Total', alpha=0.8)
ax3.bar(x, daily_stats['completion_total'], width, bottom=daily_stats['prompt_total'], label='Completion Total', alpha=0.8)
ax3.set_xlabel('Date')
ax3.set_ylabel('Total Tokens')
ax3.set_title('Daily Total Token Usage (Stacked)')
ax3.set_xticks(x)
ax3.set_xticklabels(daily_stats.index, rotation=45)
ax3.legend()

# 4. Overall totals bar
ax4.bar(['Prompt Total', 'Completion Total'], [daily_stats['prompt_total'].sum(), daily_stats['completion_total'].sum()])
ax4.set_ylabel('Grand Total Tokens')
ax4.set_title('Overall Totals Last 7 Days')

plt.tight_layout()
plt.savefig('benchmarks/token_usage_trends_last7days.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: benchmarks/token_usage_trends_last7days.png")