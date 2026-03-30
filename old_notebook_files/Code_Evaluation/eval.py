import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

#csv_file = r'C:\Users\Marcus\Desktop\Python Projs\VAA\Virtual-Academic-Advisor-with-RAG\old_notebook_files\Code_Evaluation\rag_evaluation_20251224.csv'
csv_file = r'C:\Users\Marcus\Desktop\Python Projs\VAA\Virtual-Academic-Advisor-with-RAG\old_notebook_files\Code_Evaluation\rag_evaluation_colbert_ragfusion_20260330.csv'

rows = []
system_order = []
current_system = None
header = None

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        row = [cell.strip() for cell in row]
        # Detect implementation name
        if row[0] and all(not cell for cell in row[1:]):
            current_system = row[0]
            if current_system not in system_order:
                system_order.append(current_system)
            continue
        # Detect header row
        if len(row) > 1 and row[0] == '' and row[1] == 'question':
            header = ['System'] + row[1:]
            continue
        # Data row (starts with comma)
        if len(row) > 1 and row[0] == '':
            values = [current_system] + row[1:]
            rows.append(values)

# Create DataFrame
df = pd.DataFrame(rows, columns=header)
print(df.describe())

# Convert relevant columns to numeric
df['length_word_count'] = pd.to_numeric(df['length_word_count'], errors='coerce')
df['Query time (s)'] = pd.to_numeric(df['Query time (s)'], errors='coerce')
df['retrieval_avg_similarity'] = pd.to_numeric(df['retrieval_avg_similarity'], errors='coerce')
df['retrieval_diversity'] = pd.to_numeric(df['retrieval_diversity'], errors='coerce')
df['context_utilization'] = pd.to_numeric(df['context_utilization'], errors='coerce')
df['hallucination_uncertainty_count'] = pd.to_numeric(df['hallucination_uncertainty_count'], errors='coerce')

# Group by system and compute averages
grouped = df.groupby('System', sort=False).agg({
    'length_word_count': 'mean',
    'Query time (s)': 'mean',
    'retrieval_avg_similarity': 'mean',
    'retrieval_diversity': 'mean',
    'context_utilization': 'mean',
    'hallucination_uncertainty_count': 'mean'
}).reset_index()

# Set the categorical order for plotting
grouped['System'] = pd.Categorical(grouped['System'], categories=system_order, ordered=True)
grouped = grouped.sort_values('System')

# Prepare data for grouped bar chart
systems = grouped['System']
x = np.arange(len(systems))  # the label locations
width = 0.25  # the width of the bars

# Overall metric of retrieval quality
fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width, grouped['retrieval_avg_similarity'], width, label='Avg Similarity', color='#FF6B6B', edgecolor='black')
rects2 = ax.bar(x, grouped['retrieval_diversity'], width, label='Avg Diversity', color='#F19BBF', edgecolor='black')
rects3 = ax.bar(x + width, grouped['context_utilization'], width, label='Avg Context Utilization', color='#45B7D1', edgecolor='black')

for bar in rects1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

for bar in rects2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

for bar in rects3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

# Add labels, title, legend
ax.set_ylabel('Average Value')
ax.set_xlabel('Implemented Methods')
ax.set_title('Retrieval Similarity, Diversity, and Context Utilization')
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=30)
ax.set_ylim(0,1)
ax.grid(axis='y', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('avg_retrieval_quality.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot average query time per system
plt.figure(figsize=(10,6))
data_to_plot = [df[df['System'] == s]['Query time (s)'].dropna() for s in system_order]
bp = plt.boxplot(data_to_plot, tick_labels=system_order, patch_artist=True, boxprops=dict(facecolor="#A7F579"), 
            medianprops=dict(color='red'), widths=0.5, showmeans=True, meanline=True, meanprops=dict(color='blue', linewidth=1))

# Add mean labels
means = [d.mean() for d in data_to_plot]
for i, mean in enumerate(means):
    plt.text(i + 1.28, mean, f'{mean:.1f}', ha='left', va='center', color='black', fontsize=11)

plt.ylabel('Query Time (seconds)')
plt.xlabel('Implemented Methods')
plt.title('Query Time Distribution')
plt.xticks(rotation=30)
plt.ylim(0,60)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('avg_query_time.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot average word count per system
plt.figure(figsize=(10,6))
data_to_plot = [df[df['System'] == s]['length_word_count'].dropna() for s in system_order]
bp = plt.boxplot(data_to_plot, tick_labels=system_order, patch_artist=True, boxprops=dict(facecolor="#7AD9FF"), 
            medianprops=dict(color='red'), widths=0.5, showmeans=True, meanline=True, meanprops=dict(color='blue', linewidth=1))

# Add mean labels
means = [d.mean() for d in data_to_plot]
for i, mean in enumerate(means):
    plt.text(i + 1.28, mean, f'{mean:.1f}', ha='left', va='center', color='black', fontsize=11)

plt.ylabel('Number of Words')
plt.xlabel('Implemented Methods')
plt.title('Word Count Distribution')
plt.xticks(rotation=30)
plt.ylim(0, 500)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('avg_word_count.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot average hallucination uncertainty count per system
plt.figure(figsize=(10,6))
data_to_plot = [df[df['System'] == s]['hallucination_uncertainty_count'].dropna() for s in system_order]
bp = plt.boxplot(data_to_plot, tick_labels=system_order, patch_artist=True, boxprops=dict(facecolor='#FFA07A'), 
            medianprops=dict(color='red'), widths=0.5, showmeans=True, meanline=True, meanprops=dict(color='blue', linewidth=1))

# Add mean labels
means = [d.mean() for d in data_to_plot]
for i, mean in enumerate(means):
    plt.text(i + 1.28, mean, f'{mean:.1f}', ha='left', va='center', color='black', fontsize=11)

plt.ylabel('Number of Uncertainty Words')
plt.xlabel('Implemented Methods')
plt.title('Hallucination Count Distribution (Uncertainty Words)')
plt.xticks(rotation=30)
plt.ylim(0, 3)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('avg_hallucination_count.png', dpi=300, bbox_inches='tight')
plt.show()