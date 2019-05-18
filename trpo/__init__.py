from trpo import TRPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.rllib.utils import renamed_class

TRPOAgent = renamed_class(TRPOTrainer)

__all__ = ["TRPOPolicyGraph","TRPOAgent", "APPOTrainer", "TRPOTrainer", "DEFAULT_CONFIG"]
