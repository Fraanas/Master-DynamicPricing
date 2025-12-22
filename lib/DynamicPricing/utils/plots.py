import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _preprocess_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and standardizes the baseline data.
    Ensures 'date' is datetime, maps column names, and calculates revenue.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Standardize column names (map 'sold' to 'effective_demand' if needed)
    if "effective_demand" not in df.columns and "sold" in df.columns:
        df["effective_demand"] = df["sold"]
    
    # Calculate revenue if missing
    if "revenue" not in df.columns:
        # Use .get() for safety; returns None/NaN if columns are missing
        price = df.get("sell_price", df.get("price"))
        demand = df.get("effective_demand", df.get("demand"))
        
        if price is not None and demand is not None:
            df["revenue"] = price * demand
            
    return df

def _plot_metric_comparison(ax_daily, ax_cum, agent_series, base_series, 
                            metric_name, unit_label, colors):
    """
    Plots a pair of graphs (daily and cumulative) for a specific metric.
    
    Args:
        ax_daily: Matplotlib axis for the daily plot.
        ax_cum: Matplotlib axis for the cumulative plot.
        agent_series: Time series data for the Agent.
        base_series: Time series data for the Baseline.
        metric_name: Name of the metric (e.g., "Revenue").
        unit_label: Label for the Y-axis (e.g., "Currency").
        colors: Tuple of colors (agent_color, baseline_color).
    """
    # Find common dates for accurate comparison
    common_dates = agent_series.index.intersection(base_series.index).sort_values()
    
    if common_dates.empty:
        print(f"Warning: No overlapping dates found for {metric_name}")
        return

    # Filter data to overlapping period
    ag_data = agent_series.loc[common_dates]
    ba_data = base_series.loc[common_dates]
    
    # 1. Daily Plot
    ax_daily.plot(ag_data, color=colors[0], linewidth=2, label=f'Agent {metric_name}')
    ax_daily.plot(ba_data, color=colors[1], linestyle='--', linewidth=2, label=f'Baseline {metric_name}')
    ax_daily.set_title(f"Total Daily {metric_name}", fontsize=14)
    ax_daily.set_ylabel(unit_label)
    ax_daily.legend()
    ax_daily.tick_params(axis='x', rotation=45)

    # 2. Cumulative Plot
    ax_cum.plot(ag_data.cumsum(), color=colors[0], linewidth=2, label=f'Agent Cum. {metric_name}')
    ax_cum.plot(ba_data.cumsum(), color=colors[1], linestyle='--', linewidth=2, label=f'Baseline Cum. {metric_name}')
    ax_cum.set_title(f"Cumulative {metric_name}", fontsize=14)
    ax_cum.set_ylabel(f"Cumulative {unit_label}")
    ax_cum.legend()
    ax_cum.tick_params(axis='x', rotation=45)

def _plot_price_strategy(ax, df, x_col, y_col, hue_col, title, dates_filter=None):
    """
    Plots the pricing strategy using Seaborn.
    """
    # Data aggregation: Calculate mean price per group
    avg_price = df.groupby([x_col, hue_col])[y_col].mean().reset_index()
    
    # Filter dates (if provided) to match the other charts
    if dates_filter is not None:
        avg_price = avg_price[avg_price[x_col].isin(dates_filter)]
    
    sns.lineplot(
        data=avg_price, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        ax=ax, 
        palette="tab10"
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Average Price")
    ax.tick_params(axis='x', rotation=45)
    
    # Determine legend title based on column name
    legend_title = "Product" if "id" in hue_col else "Segment"
    ax.legend(title=legend_title, loc='upper left')

def plot_market_metrics(agent_df: pd.DataFrame, base_df: pd.DataFrame):
    """
    Main function to plot the Agent vs Baseline comparison dashboard.
    Generates a 3x2 grid of plots covering Revenue, Demand, and Pricing Strategy.
    """
    # 1. Preprocessing
    base_df = _preprocess_baseline(base_df)
    
    # Prepare grouped data (Summing up daily metrics)
    agent_rev = agent_df.groupby("date")["revenue"].sum()
    base_rev = base_df.groupby("date")["revenue"].sum()
    
    agent_dem = agent_df.groupby("date")["effective_demand"].sum()
    base_dem = base_df.groupby("date")["effective_demand"].sum()

    # Identify common dates (needed for filtering price charts to match revenue/demand charts)
    common_dates = agent_rev.index.intersection(base_rev.index)

    # 2. Chart settings
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Agent vs Baseline: Market Simulation Results', fontsize=18)

    # 3. Plotting sections
    
    # --- Row 0: REVENUE ---
    _plot_metric_comparison(
        ax_daily=axs[0, 0], 
        ax_cum=axs[0, 1], 
        agent_series=agent_rev, 
        base_series=base_rev, 
        metric_name="Revenue", 
        unit_label="Currency", 
        colors=('lightblue', 'navy')
    )

    # --- Row 1: DEMAND ---
    _plot_metric_comparison(
        ax_daily=axs[1, 0], 
        ax_cum=axs[1, 1], 
        agent_series=agent_dem, 
        base_series=base_dem, 
        metric_name="Demand", 
        unit_label="Units Sold", 
        colors=('lightgreen', 'darkgreen')
    )

    # --- Row 2: PRICES ---
    # Agent Strategy (Average price per product)
    _plot_price_strategy(
        ax=axs[2, 0],
        df=agent_df,
        x_col="date",
        y_col="price",
        hue_col="product_id",
        title="Agent Strategy: Avg Price per Product"
    )

    # Baseline Strategy (Average price per product)
    _plot_price_strategy(
        ax=axs[2, 1],
        df=base_df,
        x_col="date",
        y_col="sell_price",
        hue_col="dept_id",
        title="Baseline Strategy: Avg Price per Product",
        dates_filter=common_dates # Crucial: use the same date range
    )

    plt.tight_layout()
    plt.show()

