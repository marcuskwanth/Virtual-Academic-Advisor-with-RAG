import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = '/Users/MarcussPC/Desktop/Temp/CAPSTONE/Additional_Evaluation/rag_evaluation_20251224.csv'

rows = []
system_order = []
current_system = None
header = None

with open(csv_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Detect system name (first column, rest empty)
        if ',' not in line or (line.count(',') > 5 and line.split(',')[0] and not line.split(',')[1]):
            current_system = line.split(',')[0]
            if current_system not in system_order:
                system_order.append(current_system)
            continue
        # Detect header row
        if line.startswith(',question'):
            header = ['System'] + [h.strip() for h in line.split(',')[1:]]
            continue
        # Data row (starts with comma)
        if line.startswith(','):
            values = [current_system] + [v.strip() for v in line.split(',')[1:]]
            rows.append(values)

# Create DataFrame
df = pd.DataFrame(rows, columns=header)
print(df.describe)

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
ax.set_xlabel('System')
ax.set_title('Retrieval Similarity, Diversity, and Context Utilization per System')
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=30)
ax.set_ylim(0,1)
ax.grid(axis='y', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('avg_retrieval_quality.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot average query time per system
plt.figure(figsize=(8,5))
bars2 = plt.bar(grouped['System'], grouped['Query time (s)'], color="#A7F579", edgecolor='black')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=12, rotation=0)
plt.ylabel('Average Query Time (s)')
plt.xlabel('System')
plt.title('Average Query Time per System')
plt.xticks(rotation=30)
plt.ylim(0,300)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('avg_query_time.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot average hallucination uncertainty count per system
plt.figure(figsize=(8,5))
bars3 = plt.bar(grouped['System'], grouped['hallucination_uncertainty_count'], color='#FFA07A', edgecolor='black')
for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=12, rotation=0)
plt.ylabel('Average Uncertainty Word Count')
plt.xlabel('System')
plt.title('Average Hallucination (Uncertainty Words) per System')
plt.xticks(rotation=30)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('avg_hallucination_count.png', dpi=300, bbox_inches='tight')
plt.show()