import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feature_importance_shap(model, env_eval, feature_cols, num_samples=200):
    """
    Uses SHAP to explain the agent's decisions.
    """
    # Collect a sample of data (states) from the environment
    observations = []
    
    obs, _ = env_eval.reset()
    for _ in range(num_samples):
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        observations.append(obs)
        
        # Step environment
        obs, _, terminated, truncated, _ = env_eval.step(action)
        if terminated or truncated:
            obs, _ = env_eval.reset()
            
    X_sample = np.array(observations)
    
    # Create a function wrapper for SHAP
    def model_predict_wrapper(x):
        # x is a batch of observations
        actions, _ = model.predict(x, deterministic=True)
        
        # If the action is a vector of shape (N, 1), flatten it to (N,)
        if len(actions.shape) > 1:
            return actions[:, 0]
        return actions

    # Initialize the Explainer
    # Create a background dataset (e.g., 50 k-means clusters as a reference)
    background = shap.kmeans(X_sample, 50) 
    
    explainer = shap.KernelExplainer(model_predict_wrapper, background)
    
    # calculate SHAP values for our sample
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_sample)
    
    # Plot the summary chart
    plt.figure()
    plt.title("Feature Impact on Agent's Pricing Decisions")
    
    # Check if shap_values is a list (often happens with KernelExplainer) and take the first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols)