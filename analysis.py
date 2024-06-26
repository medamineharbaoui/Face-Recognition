import json
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from main import recognition_t, confidence_t 

# Function to load results from a file
def load_results(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results

# Function to calculate confusion matrix metrics
def calculate_confusion_matrix_metrics(true_labels, predicted_labels, target_name):
    tp = np.sum((true_labels == predicted_labels) & (true_labels == target_name))
    fp = np.sum((true_labels != predicted_labels) & (predicted_labels == target_name))
    tn = np.sum((true_labels == predicted_labels) & (true_labels != target_name))
    fn = np.sum((true_labels != predicted_labels) & (predicted_labels != target_name))
    return tp, fp, tn, fn

# Function to calculate TPR and FPR
def calculate_tpr_fpr(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity or True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    return tpr, fpr

# Function to load or initialize metrics list from a JSON file
def load_or_initialize_metrics(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Function to save metrics list to a JSON file
def save_metrics(file_path, metrics_list):
    with open(file_path, 'w') as f:
        json.dump(metrics_list, f, indent=4)

# Function to calculate various metrics and return them
def calculate_metrics(true_labels, predicted_labels, target_name):
    tp, fp, tn, fn = calculate_confusion_matrix_metrics(true_labels, predicted_labels, target_name)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity or Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    tpr, fpr = calculate_tpr_fpr(tp, fp, tn, fn)

    print('----------------------')
    print('True Positives : ', tp)
    print('False Positives : ', fp)
    print('True Negatives : ', tn)
    print('False Negatives : ', fn)
    print('----------------------')

    return accuracy, precision, sensitivity, specificity, f1, tpr, fpr


# Load the results from the main script
results_store = load_results('results_store.pkl')

# Set the name of the person you are analyzing
person_name = "Harbaoui"

# Define true labels based on the images in the Faces directory for the given person
true_labels = []  # This should include known and unknown labels appropriately
person_dir = os.path.join('Faces', person_name)
for image_name in os.listdir(person_dir):
    true_labels.append(person_name)
    
# Add more labels to simulate a realistic dataset
true_labels.extend(['unknown'] * (len(results_store) - len(true_labels)))

# Ensure true_labels has the same length as results_store
true_labels = true_labels[:len(results_store)]

# Extract predicted labels from the result store
predicted_labels = [name if name == person_name else 'unknown' for name, _, _ in results_store]

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate the metrics for the current evaluation
accuracy, precision, sensitivity, specificity, f1, tpr, fpr = calculate_metrics(true_labels, predicted_labels, person_name)

# Print the calculated metrics
print('----------------------')
print(f"Metrics for '{person_name}' with Confidence_t = {confidence_t} and Recognition_t = {recognition_t}:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
#print(f"F1 Score: {f1:.2f}")
print(f"True Positive Rate (TPR): {tpr:.2f}")
print(f"False Positive Rate (FPR): {fpr:.2f}")
print('----------------------')

# Initialize or load the metrics list from the JSON file
metrics_list = load_or_initialize_metrics('metrics.json')

# Append the current evaluation metrics to the list
metrics_list.append({
    "Thresholds": (confidence_t, recognition_t),
    "TPR": tpr,
    "FPR": fpr
})

# Save the updated metrics list to the JSON file
save_metrics('metrics.json', metrics_list)

print("Metrics have been saved to metrics.json")
