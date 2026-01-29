import os
import matplotlib.pyplot as plt
import pandas as pd

def _run_single_episode(env, model) -> None:
    """
    Executes a single simulation episode (product/segment combination).
    """
    obs, _ = env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

def _save_plot(env, output_dir: str, episode_idx: int) -> None:
    """
    Renders the environment state and saves the plot to a file.
    """

    env.render()
    
    # Create a descriptive filename
    filename = f"eval_{episode_idx}_{env.current_product}_{env.current_customer}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath)
    plt.close() # Close figure to free memory

def _process_episode_data(env) -> pd.DataFrame:
    """
    Converts environment history into a structured DataFrame with metadata.
    """
    df_episode = pd.DataFrame(env.history)
    df_episode["product_id"] = env.current_product
    df_episode["segment_id"] = env.current_customer
    return df_episode

def run_evaluation(env, model, output_dir: str = "./plots/") -> pd.DataFrame:
    """
    Main orchestration function for running full evaluation.
    
    Args:
        env: The Gym environment.
        model: The trained RL agent.
        output_dir: Directory to save evaluation plots.
        
    Returns:
        pd.DataFrame: Aggregated results from all episodes.
    """
    # 1. Setup
    env = env.envs[0]
    os.makedirs(output_dir, exist_ok=True)
    all_history = []
    
    # Ensure we start from the first combination
    if hasattr(env, 'eval_idx'):
        env.eval_idx = 0 
        
    print(f"Starting evaluation for {env.n_combinations} combinations...")

    # 2. Main Evaluation Loop
    for i in range(env.n_combinations):
        # A. Run Simulation
        _run_single_episode(env, model)
        
        # B. Visualize & Save
        _save_plot(env, output_dir, episode_idx=i)
        
        # C. Collect Data
        episode_df = _process_episode_data(env)
        all_history.append(episode_df)

    # 3. Aggregation & Formatting
    full_results_df = pd.concat(all_history, ignore_index=True)
    full_results_df["date"] = pd.to_datetime(full_results_df["date"])
    
    print(f"Evaluation finished. Plots saved to {output_dir}")
    return full_results_df