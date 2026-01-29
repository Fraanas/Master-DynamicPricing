import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Any, Dict, Mapping, Optional, Literal

from lib.utils.transformation import transformation
from lib.utils.revenue_weights import RevenueWeights



class DynamicPricingEnv(Env):
    """
    Each episode represents pricing decisions for a single product–customer segment
    combination over a sequence of days. The agent adjusts prices daily in order
    to maximize revenue under price elasticity and demand uncertainty.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        eval_start: str,
        eval_end: str,
        # Demand and Reve parameters
        price_change_limit: float = 0.15,
        price_elasticity: float = -1.5,
        penalty: float = 0.02,
        mode: Literal['train','eval']='train',
        revenue_weights: Optional[RevenueWeights] = None,
        stochastic_elasticity: bool | None = None,
        random_start: bool | None = None,
        seed: int | None = None,
        render_mode=None
    ):
        """
        Initializes the dynamic pricing environment.
        - Splits the input dataset into training and evaluation periods.
        - Defines the action space as a continuous price adjustment in [-1, 1],
        scaled by a maximum daily price change limit.
        - Defines the observation space based on lagged prices, demand signals,
        and one-hot encoded product and customer segment identifiers.
        - Prepares all possible product–customer segment combinations that can
        be sampled as independent episodes.
        """
        super().__init__()

        # basic set up
        self.rng = np.random.default_rng(seed)
        self.price_change_limit = price_change_limit
        self.price_elasticity = price_elasticity
        self.penalty = penalty
        self.render_mode = render_mode
        self.mode = mode
        self.revenue_weights = revenue_weights
        self.current_weight = 1.0
        self.episode_count = 0
        self.reset_count = 0

        self.epsilon = 1e-6
        self.elasticity_std = 0.15        # Elastisity noise 
        self.min_price_ratio = 0.5  
        self.max_price_ratio = 2.0       
        self.max_demand_cap = 1.2         # Max  demand increase is (1.X) to +X% 

        self.reward_scale = 10.0

        self.stochastic_elasticity = (mode == "train") if stochastic_elasticity is None else stochastic_elasticity
        self.random_start = (mode == "train") if random_start is None else random_start


        # split data 
        self.df_train, self.df_eval, self.ohe_cols = transformation(df, start_date, end_date, eval_start, eval_end)
        self.df = (self.df_train if mode == "train" else self.df_eval).reset_index(drop=True)

        # Action Space: Single continuous value in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Define feature columns for observation
        self.feature_cols = [     
            "sell_price_l_1",       
            "yhat_sold",
            "sold_l_1",
            "dow_sin",
            "dow_cos",
            "week_of_year_sin",
            "week_of_year_cos",
            "revenue_ma_7",  
            "revenue_ma_28",          
            "sold_ma_7",          
            "sold_ma_28",                  
            "price_change_l_1",
            "price_delta"
        ] + self.ohe_cols          

        '''
        # Define feature columns for observation
        self.feature_cols = [       # TODO: Cols to think: is_weekend, is_holiday, 
            "sell_price_l_1",       # Product Price from previous day
            "demand_l_1",           # Demand from previous day
            "demand_ma_7",          # Average demand from last week
            "demand_l_365",         # Demand from previous year
            "yhat_demand",          # Forecasted todays demand
            "demand_delta_1",       # Diffenrence beetween demand->demand_l_1 - demand_l_2
            "dow_sin",              # Sinus day of the week (0-6) 
            "dow_cos" ,             # Cosinus day of the week (0-6) 
            "week_of_year_sin",
            "week_of_year_cos",
        ] + self.ohe_cols           # Product and customer segment columns
        '''
                            
        self.obs_dim = len(self.feature_cols) 

        # Observation Space: Continuous values for features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim, ), dtype=np.float32
        )

        # List of combinations product x customer segment
        self.combinations = (
            self.df[['dept_id', 'customer_segment']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.n_combinations = len(self.combinations)

        self.state = None
        self.current_step = None
        
        

    def reset(self, seed=None, options=None):
        """
        Resets the environment at the beginning of a new episode.
        Each episode:
        - Randomly selects one product–customer segment combination.
        - Extracts the corresponding historical time series.
        - Randomly selects a starting day sufficiently far from the beginning
        to ensure availability of lagged features.
        - Initializes lag-based features so that the agent observes the previous
        day's price before making its first decision.
        Returns:
            state (np.ndarray): Initial observation vector for the agent.
            info (dict): Empty dictionary (Gym API compatibility).
        """
        super().reset(seed=seed)

        # set random combination of product and customer segment 
        idx = self.rng.integers(0, self.n_combinations)
        row = self.combinations.iloc[idx]
        self.current_product = row["dept_id"]
        self.current_customer = row["customer_segment"]

        # use data only for this combination
        self.current_df = self.df[
            (self.df["dept_id"] == self.current_product) &
            (self.df["customer_segment"] == self.current_customer)
        ].reset_index(drop=True).copy()

        self.reset_count += 1
        self.episode_count += 1

        # set weights for each combination 
        if self.revenue_weights is None:
            self.current_weight = 1.0
        else:
            self.current_weight = self.revenue_weights.get(self.current_product, self.current_customer)
        
        # Lag bufor, min number of days = 365 days
        min_start = min(1, len(self.current_df) // 2)
        max_start = len(self.current_df) - 1
        if max_start <= min_start:
            raise ValueError("Not enough data for this product-segment combination")
        
        # set a start 
        if self.random_start:
            self.current_step = self.rng.integers(min_start, max_start)
        else:
            self.current_step = min_start

        # price delta spot
        self.current_df["price_change_l_1"] = 0.0
        self.current_df["price_delta"] = 0.0
        self.current_df.loc[self.current_step, "sell_price_l_1"] = self.current_df.loc[self.current_step - 1, "sell_price"]

        # build current state
        self.state = (
            self.current_df
            .loc[self.current_step, self.feature_cols]
            .values
            .astype(np.float32)
        )
        # log previous episode BEFORE resetting history
        if hasattr(self, "logger") and hasattr(self, "history"):
            if len(self.history["revenue"]) > 0:
                self.logger.record(
                    "episode/cumulative_revenue",
                    np.sum(self.history["revenue"])
                )

        # reset history for new episode
        self.history = {
            "date": [],     
            "day": [],
            "price": [],
            "price_delta":[],
            "effective_demand": [],
            "revenue": [],
            "action": []
        }

        return self.state, {}
    

    def _calculate_demand(self, baseline_demand, price, prev_price): 
        price_ref = max(prev_price, self.epsilon)

        # Stochastic elasticity
        elasticity = self.price_elasticity
        if self.stochastic_elasticity:
            elasticity *= self.rng.normal(1.0, self.elasticity_std) 

        price_ratio = np.clip(price / price_ref, self.min_price_ratio, self.max_price_ratio) 

        calculated_demand = baseline_demand * (price_ratio ** elasticity) 
        effective_demand = min(calculated_demand, self.max_demand_cap * baseline_demand)
        effective_demand = max(effective_demand, 0.0)
        
        return effective_demand
    
    def _calculate_reward(self, price, demand, prev_price):
        agent_revenue = price * demand
    
        baseline_demand = self._calculate_demand(
            baseline_demand=self.current_df.loc[self.current_step, "yhat_sold"],
            price=prev_price,
            prev_price=prev_price
        )
        baseline_revenue = prev_price * baseline_demand
        
        diff = agent_revenue - baseline_revenue
        weighted_reward = diff * self.current_weight

        # Penalty for instability
        if self.penalty > 0:
            change_magnitude = abs(price / max(prev_price, self.epsilon) - 1.0)
            penalty_val = self.penalty * change_magnitude * baseline_revenue * self.current_weight
            weighted_reward -= penalty_val
        
        return weighted_reward / self.reward_scale


    def step(self, action):
        """
        Executes one environment step (one day).
        Process:
            - Converts the agent's continuous action into a bounded price change.
            - Updates the current day's price based on the agent's decision.
            - Computes effective demand using a price elasticity model with stochastic noise.
            - Calculates reward as revenue minus a penalty for large price changes.
            - Propagates the effects of the pricing decision to the next day's lagged features
            (price, price change, and realized demand).
            - Advances the environment to the next day.
        Args:
            action (np.ndarray): Continuous action in [-1, 1] representing relative price adjustment.
        Returns:
            next_state (np.ndarray): Observation vector for the next day.
            reward (float): Normalized reward (daily revenue minus penalty).
            terminated (bool): Whether the episode has ended.
            truncated (bool): Always False (no artificial truncation).
            info (dict): Empty dictionary (Gym API compatibility).
        """
        terminated = False
        truncated = False

        # set current step
        row = self.current_df.loc[self.current_step]
    
        prev_price = row["sell_price"] 

        # action ∈ [-1, 1] -> new price
        a = float(action[0])

        # map to price change
        delta = np.clip(a, -1.0, 1.0) * self.price_change_limit
        price = prev_price * (1.0 + delta)

        # upload price 
        self.current_df.loc[self.current_step, "sell_price"] = price

        # todays demand based on forecast
        baseline_demand = row["yhat_sold"] # yhat_demand
        price_ref = prev_price
        price_ref = max(price_ref, 1e-6)

        # calculate price change 
        price_change_1 = price / price_ref - 1.0
        self.current_df.loc[self.current_step, "price_change_l_1"] = price_change_1
        self.current_df.loc[self.current_step, "price_delta"] = delta

        # calculate price elasiscity
        effective_demand = self._calculate_demand(baseline_demand, price, prev_price)

        # Reward -> revenue
        norm_reward = self._calculate_reward(price, effective_demand, prev_price) 

        # advance time
        t = self.current_step
        dt = self.current_df.loc[t, "date"] 

        # push action effect into next day's lag feature
        next_step =  t + 1

        terminated = next_step >= len(self.current_df)
        truncated = False

        if not terminated:
            self.current_df.loc[next_step, "sell_price_l_1"] = price
            self.current_df.loc[next_step, "price_change_l_1"] = price_change_1
            self.current_df.loc[next_step, "price_delta"] = delta
            self.current_df.loc[next_step, "sold_l_1"] = effective_demand
        
            self.current_step = next_step
            self.state = (
                self.current_df.loc[self.current_step, self.feature_cols]
                .values.astype(np.float32)
            )
        else:
            pass

        # keep data 
        self.history["date"].append(pd.to_datetime(dt)) 
        self.history["day"].append(t)
        self.history["price"].append(price)
        self.history["price_delta"].append(delta)
        self.history["effective_demand"].append(effective_demand)
        self.history["revenue"].append(price * effective_demand)
        self.history["action"].append(a)

        # log values to tensorboard
        if hasattr(self, "logger"):
            self.logger.record("pricing/price", price)
            self.logger.record("pricing/effective_demand", effective_demand)
            self.logger.record("pricing/revenue", price * effective_demand)
            self.logger.record("pricing/price_delta", delta)
            self.logger.record("episode/avg_price", np.mean(self.history["price"]))
        else:
            pass

        return self.state, norm_reward, terminated, truncated, {}
        


    def render(self):
        """
        Visualizes pricing policy and revenue over the episode.
        """

        if not hasattr(self, "history") or len(self.history["day"]) == 0:
            print("Nothing to render yet.")
            return

        dates = self.history["date"]
        prices = self.history["price"]
        demand = self.history["effective_demand"]
        revenue = self.history["revenue"]
        actions = self.history["action"]

        cum_revenue = np.cumsum(revenue)

        fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
        combo = f"combo=({self.current_product}, {self.current_customer})"
        fig.suptitle(f"{combo}", fontsize=12)

        # 1. Price
        axs[0].plot(dates, prices, label="Price", color="lightblue")
        axs[0].set_ylabel("Price")
        axs[0].set_title("Price trajectory")
        axs[0].grid(True)
    
        # 2. Demand
        axs[1].plot(dates, demand, label="Demand", color="orange")
        axs[1].set_ylabel("Demand")
        axs[1].set_title("Demand response")
        axs[1].grid(True)
        
        # 3. Daily revenue
        axs[2].plot(dates, revenue, label="Revenue", color="navy")
        axs[2].set_ylabel("Revenue")
        axs[2].set_title("Daily revenue")
        axs[2].grid(True)

        # 4. Cumulative revenue
        axs[3].plot(dates, cum_revenue, label="Cumulative revenue", color="red")
        axs[3].set_ylabel("Cum. revenue")
        axs[3].set_title("Cumulative revenue")
        axs[3].set_xlabel("Date")
        fig.autofmt_xdate()
        axs[3].grid(True)

        # 5. actions
        axs[4].plot(dates, actions, label="Action", color="green")
        axs[4].set_ylabel("Actions")
        axs[4].set_title("Daily actions")
        axs[4].grid(True)


        plt.tight_layout()
        #plt.show()