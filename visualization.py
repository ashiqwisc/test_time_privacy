import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# Save visualization of softmax outputs
def save_softmax_visualizations(all_softmax_outputs, all_labels, num_classes, img_save_path, model_type=None):
    print("here")
    """
    Save visualizations of softmax outputs.
    
    Args:
        all_softmax_outputs: Array of softmax outputs for all samples
        all_labels: Array of true labels
        num_classes: Number of classes in the dataset
        img_save_path: Base path to save images
        model_type: Type of model ('pretrained_model', 'warmup_model', 'certified_model', 'retrain_model', 'pareto_model' etc.)
    """
    if img_save_path:
        print(img_save_path)
        print("model type: ", model_type)
        # make a new directory with model type 
        if model_type in ["pretrained_model", "warmup_model", "certified_model", "retrain_model", "pareto_model"]:
            img_save_path = os.path.join(img_save_path, f"{model_type}_outputs")
            
        os.makedirs(img_save_path, exist_ok=True)

        # save raw softmax outputs 
        softmax_output_path = os.path.join(img_save_path, "softmax_outputs.csv")
        np.savetxt(softmax_output_path, all_softmax_outputs, delimiter=',')
        
        # save first 20 visual sample of softmax outputs 
        num_samples = min(20, len(all_softmax_outputs))
        
        for i in range(num_samples):
            plt.figure(figsize=(10, 4))
            
            # Plot softmax output (bar chart)
            plt.subplot(1, 2, 1)
            plt.bar(range(num_classes), all_softmax_outputs[i])
            plt.title(f"Sample {i+1} - True Label: {all_labels[i]}")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            
            # Plot difference from uniform distribution
            plt.subplot(1, 2, 2)
            diff_from_uniform = all_softmax_outputs[i] - (1/num_classes)
            plt.bar(range(num_classes), diff_from_uniform)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title("Difference from Uniform")
            plt.xlabel("Class")
            plt.ylabel("Difference")
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_save_path, f"sample_{i+1}_output.pdf"), format='pdf')
            plt.close()
        
        # Visualization showing average softmax output across all samples
        plt.figure(figsize=(10, 6))
        avg_softmax = np.mean(all_softmax_outputs, axis=0)
        plt.bar(range(num_classes), avg_softmax)
        plt.axhline(y=1/num_classes, color='r', linestyle='--', label="Uniform")
        plt.title(f"Average Softmax Output Across All Forget Samples{' (' + model_type + ')' if model_type else ''}")
        plt.xlabel("Class")
        plt.ylabel("Average Probability")
        plt.legend()
        plt.savefig(os.path.join(img_save_path, "average_softmax_output.pdf"), format='pdf')
        plt.close()