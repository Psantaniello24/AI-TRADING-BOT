from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os

class TradingRLAgent:
    def __init__(self, env, model_name="ppo_trading", save_path="./models"):
        """
        Initialize the RL agent
        
        Args:
            env: Trading environment
            model_name (str): Name of the model
            save_path (str): Path to save model checkpoints
        """
        self.env = env
        self.model_name = model_name
        self.save_path = save_path
        
        # Create directory for saving models if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Vectorize the environment
        self.vec_env = DummyVecEnv([lambda: env])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
        
        # Initialize the PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=f"./logs/{model_name}"
        )
        
    def train(self, total_timesteps=100000, eval_freq=10000):
        """
        Train the RL agent
        
        Args:
            total_timesteps (int): Total number of timesteps to train for
            eval_freq (int): Frequency of evaluation
            
        Returns:
            PPO: Trained model
        """
        # Custom callback for evaluation
        eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path=f"{self.save_path}/best",
            log_path=f"{self.save_path}/results",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.model.save(f"{self.save_path}/{self.model_name}")
        
        return self.model
    
    def load(self, model_path=None):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the trained model
            
        Returns:
            PPO: Loaded model
        """
        if model_path is None:
            model_path = f"{self.save_path}/{self.model_name}"
            
        self.model = PPO.load(model_path, self.vec_env)
        return self.model
    
    def predict(self, observation, deterministic=True):
        """
        Make a prediction with the trained model
        
        Args:
            observation: Current observation
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            tuple: (action, _)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action, _
    
    def test(self, env, num_episodes=1):
        """
        Test the agent on the environment
        
        Args:
            env: Environment to test on
            num_episodes (int): Number of episodes to run
            
        Returns:
            list: List of episode rewards
        """
        episode_rewards = []
        
        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
            episode_rewards.append(total_reward)
            print(f"Episode {i+1}: Total Reward: {total_reward}")
            
        return episode_rewards
        
    def get_explanation(self, observation):
        """
        Get explanation for trading decision
        
        Args:
            observation: Current observation
            
        Returns:
            dict: Explanation including action, probabilities, and reasoning
        """
        action, _ = self.model.predict(observation, deterministic=False)
        
        # Get action probabilities
        action_probs = self.model.policy.get_distribution(observation).distribution.probs.detach().numpy()
        
        # Map actions to labels
        action_labels = {0: "Hold", 1: "Buy", 2: "Sell"}
        chosen_action = action_labels[action.item()]
        
        # Create explanation
        explanation = {
            "action": chosen_action,
            "action_id": action.item(),
            "probabilities": {
                action_labels[i]: float(prob) 
                for i, prob in enumerate(action_probs[0])
            },
            "confidence": float(action_probs[0][action.item()]),
            "reasoning": self._generate_reasoning(observation, action.item(), action_probs[0])
        }
        
        return explanation
    
    def _generate_reasoning(self, observation, action, probabilities):
        """
        Generate a reasoning for the action taken
        
        Args:
            observation: Current observation
            action (int): Action taken
            probabilities (np.array): Action probabilities
            
        Returns:
            str: Reasoning for the action
        """
        # This is a simplified reasoning. In a production system, you might want to
        # analyze the policy network outputs or use explainable AI techniques.
        
        confidence = probabilities[action]
        
        if action == 0:  # Hold
            if confidence > 0.8:
                return "High confidence in holding as the current trend is uncertain."
            else:
                return "Slightly favored holding over other actions due to market conditions."
        elif action == 1:  # Buy
            if confidence > 0.8:
                return "Strong buying signal detected, high confidence in positive price movement."
            else:
                return "Modest buying opportunity detected based on recent trends."
        elif action == 2:  # Sell
            if confidence > 0.8:
                return "Strong selling signal detected, high confidence in negative price movement."
            else:
                return "Modest selling opportunity based on current market conditions."
                
        return "No clear reasoning available."

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging metrics during training
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                
                # Log additional metrics
                if 'net_worth' in info:
                    self.logger.record('trading/net_worth', info['net_worth'])
                if 'profit_pct' in info:
                    self.logger.record('trading/profit_pct', info['profit_pct'])
                if 'sharpe_ratio' in info:
                    self.logger.record('trading/sharpe_ratio', info['sharpe_ratio'])
                    
        return True 