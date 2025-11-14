"""
Strategy Configuration

Define strategies in YAML/JSON config files with:
- Strategy metadata (name, description, version)
- Market data subscriptions
- Trading parameters
- Risk limits
- Initial positions (for resuming)

Example config:
  name: "BTC_MA_Cross"
  version: "1.0.0"
  description: "BTC moving average crossover strategy"

  subscriptions:
    exchanges: ["binance"]
    symbols: ["BTC/USDT"]
    data_types: ["tick", "orderbook"]

  parameters:
    fast_period: 10
    slow_period: 30
    position_size: 0.1

  risk_limits:
    max_position_size: 1.0
    max_daily_loss: 1000
    max_drawdown_pct: 10

  initial_positions_file: "./positions/btc_strategy.csv"
"""
import yaml
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from loguru import logger


@dataclass
class SubscriptionConfig:
    """Market data subscription configuration"""
    exchanges: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=lambda: ["tick"])  # tick, trade, orderbook, bar
    bar_interval: Optional[str] = None  # For bars: 1m, 5m, 1h, etc


@dataclass
class RiskLimitsConfig:
    """Risk limits configuration"""
    max_position_size: Optional[Decimal] = None
    max_order_size: Optional[Decimal] = None
    max_daily_loss: Optional[Decimal] = None
    max_drawdown_pct: Optional[Decimal] = None
    max_concentration_pct: Optional[Decimal] = None


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration

    Loaded from YAML/JSON file
    """

    # Metadata
    name: str
    version: str = "1.0.0"
    description: str = ""
    strategy_id: Optional[str] = None  # Unique ID (auto-generated if not provided)

    # Subscriptions (legacy - use universe instead)
    subscriptions: SubscriptionConfig = field(default_factory=SubscriptionConfig)

    # Universe configuration (recommended way to define instruments)
    universe: Optional[Dict] = None  # Universe configuration

    # Strategy parameters (custom per strategy)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Risk limits
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)

    # Persistence
    initial_positions_file: Optional[str] = None  # File to load initial positions from
    save_state_on_stop: bool = True  # Auto-save state on stop
    state_file: Optional[str] = None  # Where to save state

    # Execution
    enabled: bool = True  # Can be disabled without removing config
    paper_trading: bool = False  # Paper trading mode
    max_orders_per_minute: int = 60  # Rate limiting

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_file(cls, filepath: str) -> "StrategyConfig":
        """
        Load configuration from file

        Supports YAML and JSON formats.

        Args:
            filepath: Path to config file

        Returns:
            StrategyConfig instance
        """

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        # Load file
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        logger.info(f"Loaded strategy config from {filepath}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "StrategyConfig":
        """Create config from dictionary"""

        # Parse subscriptions
        sub_data = data.get("subscriptions", {})
        subscriptions = SubscriptionConfig(**sub_data)

        # Parse risk limits (convert strings to Decimal)
        risk_data = data.get("risk_limits", {})
        risk_limits = RiskLimitsConfig(
            max_position_size=Decimal(str(risk_data["max_position_size"])) if "max_position_size" in risk_data else None,
            max_order_size=Decimal(str(risk_data["max_order_size"])) if "max_order_size" in risk_data else None,
            max_daily_loss=Decimal(str(risk_data["max_daily_loss"])) if "max_daily_loss" in risk_data else None,
            max_drawdown_pct=Decimal(str(risk_data["max_drawdown_pct"])) if "max_drawdown_pct" in risk_data else None,
            max_concentration_pct=Decimal(str(risk_data["max_concentration_pct"])) if "max_concentration_pct" in risk_data else None,
        )

        # Generate strategy_id if not provided
        strategy_id = data.get("strategy_id") or f"{data['name']}_{data.get('version', '1.0.0')}"

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            strategy_id=strategy_id,
            subscriptions=subscriptions,
            universe=data.get("universe"),  # Universe configuration
            parameters=data.get("parameters", {}),
            risk_limits=risk_limits,
            initial_positions_file=data.get("initial_positions_file"),
            save_state_on_stop=data.get("save_state_on_stop", True),
            state_file=data.get("state_file"),
            enabled=data.get("enabled", True),
            paper_trading=data.get("paper_trading", False),
            max_orders_per_minute=data.get("max_orders_per_minute", 60),
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
        )

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        data = asdict(self)

        # Convert Decimals to strings
        risk_limits = data.get("risk_limits", {})
        for key, value in risk_limits.items():
            if isinstance(value, Decimal):
                risk_limits[key] = str(value)

        return data

    def to_file(self, filepath: str, format: str = "yaml") -> None:
        """
        Save configuration to file

        Args:
            filepath: Output file path
            format: Format ('yaml' or 'json')
        """

        data = self.to_dict()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            elif format == "json":
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved strategy config to {filepath}")

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid, raises ValueError if not
        """

        if not self.name:
            raise ValueError("Strategy name is required")

        if not self.strategy_id:
            raise ValueError("Strategy ID is required")

        if not self.subscriptions.symbols:
            logger.warning(f"Strategy {self.name} has no symbols subscribed")

        return True

    def get_state_file(self) -> str:
        """Get state file path (auto-generate if not specified)"""
        if self.state_file:
            return self.state_file

        # Auto-generate from strategy_id
        return f"./data/strategy_state/{self.strategy_id}.json"

    def get_positions_file(self) -> Optional[str]:
        """Get initial positions file path"""
        return self.initial_positions_file


def create_example_config(name: str = "Example_Strategy") -> StrategyConfig:
    """Create an example configuration"""

    return StrategyConfig(
        name=name,
        version="1.0.0",
        description="Example strategy configuration",
        subscriptions=SubscriptionConfig(
            exchanges=["binance", "simulated"],
            symbols=["BTC/USDT", "ETH/USDT"],
            data_types=["tick", "orderbook"],
        ),
        parameters={
            "fast_period": 10,
            "slow_period": 30,
            "position_size": 0.1,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
        },
        risk_limits=RiskLimitsConfig(
            max_position_size=Decimal("10"),
            max_order_size=Decimal("5"),
            max_daily_loss=Decimal("1000"),
            max_drawdown_pct=Decimal("10"),
            max_concentration_pct=Decimal("30"),
        ),
        initial_positions_file=None,
        enabled=True,
        paper_trading=False,
    )
