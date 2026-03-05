import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_plots(model, X_sample, feature_names, output_path):
    # Use a background summary for faster SHAP calculation
    background = X_sample[:50] 
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_sample[:10])
    
    plt.figure(figsize=(10, 6))
    # Reshape back to 2D for the summary plot
    shap.summary_plot(shap_values[0], X_sample[:10].reshape(10, -1), 
                      feature_names=feature_names, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()