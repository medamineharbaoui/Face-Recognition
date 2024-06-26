import json
import matplotlib.pyplot as plt

# Function to load metrics from JSON file
def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics_list = json.load(f)
    return metrics_list

# Function to plot ROC curve
def plot_roc_curve(metrics_list):
    # Sort metrics based on recognition_t for correct curve plotting
    metrics_list_sorted = sorted(metrics_list, key=lambda x: x['Thresholds'][1])  # Sort by recognition_t
    
    # Extract TPR and FPR values
    tprs = [entry['TPR'] for entry in metrics_list_sorted]
    fprs = [entry['FPR'] for entry in metrics_list_sorted]
    thresholds = [entry['Thresholds'] for entry in metrics_list_sorted]  # List of (confidence_t, recognition_t) pairs
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, marker='o', linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')  # Diagonal line for random classifier
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.axis([0, 1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Annotate each point with (confidence_t, recognition_t)
    for i, txt in enumerate(thresholds):
        plt.annotate(f"{txt}", (fprs[i], tprs[i]), textcoords="offset points", xytext=(5,-5), ha='left')
    
    plt.show()
# Load metrics data from JSON file
metrics_list = load_metrics('metrics.json')

# Plot ROC curve
plot_roc_curve(metrics_list)
