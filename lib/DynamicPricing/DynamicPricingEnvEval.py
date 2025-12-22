import numpy as np
from lib.DynamicPricing.DynamicPricingEnv import DynamicPricingEnv

class DynamicPricingEnvEval(DynamicPricingEnv):
    """
    Deterministic evaluation environment.
    Same MDP, different stochasticity and reset policy.
    """
    def __init__(self, **kwargs):
        # drop the keys we want to overwrite 
        for key in ['stochastic_elasticity', 'random_start', 'mode', 'seed']:
            kwargs.pop(key, None)
        # initialize a parent with eval parameters
        super().__init__(
            stochastic_elasticity=False,
            random_start=False,
            mode='eval',
            seed=42,
            **kwargs
        )
        # additional counter 
        self.eval_idx = 0

    def reset(self, seed=None, options=None):
        """
        Overries reset to select combinations sequentially (0, 1, 2...) instead of randomly.
        """
        super().reset(seed=seed) 
        
        idx = self.eval_idx % self.n_combinations
        
        row = self.combinations.iloc[idx]
        self.current_product = row["dept_id"]
        self.current_customer = row["customer_segment"]
        
        self.current_df = self.df[
            (self.df["dept_id"] == self.current_product) &
            (self.df["customer_segment"] == self.current_customer)
        ].reset_index(drop=True).copy()

        self.reset_count += 1
        self.episode_count += 1
        
        if self.revenue_weights is None:
            self.current_weight = 1.0
        else:
            self.current_weight = self.revenue_weights.get(self.current_product, self.current_customer)

        min_start = min(1, len(self.current_df) // 2)
        self.current_step = min_start 

        # lag initialization 
        self.current_df["price_change_l_1"] = 0.0
        self.current_df["price_delta"] = 0.0
        self.current_df.loc[self.current_step, "sell_price_l_1"] = self.current_df.loc[self.current_step - 1, "sell_price"]

        # build state
        self.state = (
            self.current_df
            .loc[self.current_step, self.feature_cols]
            .values
            .astype(np.float32)
        )
        
        # history reset
        self.history = {
            "date": [], "day": [], "price": [], "price_delta": [], "effective_demand": [], "revenue": [], "action": [],
        }

        #print(f"[EVAL] Testing combo {idx+1}/{self.n_combinations}: ({self.current_product}, {self.current_customer})")

        # next product-segment combination
        self.eval_idx += 1
        
        return self.state, {}
    

    
