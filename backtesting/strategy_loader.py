"""
Strategy Loader

Dynamically loads strategy code for backtesting.
"""
from typing import Type, Any, Optional
import sys
import importlib.util
import tempfile
import os
from pathlib import Path
from loguru import logger


class StrategyLoader:
    """
    Dynamically loads strategy classes from code strings

    Single Responsibility: Load and validate strategy code
    """

    @staticmethod
    def load_from_code(code: str, class_name: Optional[str] = None) -> Type:
        """
        Load a strategy class from code string

        Args:
            code: Python code as string
            class_name: Name of the strategy class to load (auto-detect if None)

        Returns:
            Strategy class

        Raises:
            ValueError: If code is invalid or class not found
        """
        # Create temporary module
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "temp_strategy.py")

        try:
            # Write code to temp file
            with open(temp_file, 'w') as f:
                # Always add necessary imports first (before anything else)
                f.write("import pandas as pd\n")
                f.write("import numpy as np\n")
                f.write("from typing import Dict, Any\n\n")

                # Add BaseStrategy if not present
                if "class BaseStrategy" not in code:
                    f.write("""
class BaseStrategy:
    \"\"\"Base class for trading strategies\"\"\"

    def __init__(self, name: str = "Strategy", parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Calculate indicators needed for the strategy\"\"\"
        raise NotImplementedError

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Generate trading signals (1=buy, -1=sell, 0=hold)\"\"\"
        raise NotImplementedError

""")

                f.write(code)

            # Load module
            spec = importlib.util.spec_from_file_location("temp_strategy", temp_file)
            if spec is None or spec.loader is None:
                raise ValueError("Failed to load strategy module")

            module = importlib.util.module_from_spec(spec)
            sys.modules["temp_strategy"] = module
            spec.loader.exec_module(module)

            # Find the strategy class
            if class_name is None:
                # Auto-detect: find first class that's not BaseStrategy
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and
                        name != 'BaseStrategy' and
                        name.endswith('Strategy')):
                        class_name = name
                        break

            if class_name is None:
                raise ValueError("No strategy class found in code")

            # Get the class
            if not hasattr(module, class_name):
                raise ValueError(f"Class {class_name} not found in code")

            strategy_class = getattr(module, class_name)

            # Validate it has required methods
            if not hasattr(strategy_class, 'calculate_indicators'):
                raise ValueError(f"{class_name} missing calculate_indicators method")

            if not hasattr(strategy_class, 'generate_signals'):
                raise ValueError(f"{class_name} missing generate_signals method")

            logger.debug(f"Successfully loaded strategy class: {class_name}")
            return strategy_class

        except Exception as e:
            logger.error(f"Failed to load strategy: {e}")
            raise

        finally:
            # Cleanup
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass

    @staticmethod
    def load_from_file(file_path: str, class_name: Optional[str] = None) -> Type:
        """
        Load a strategy class from file

        Args:
            file_path: Path to Python file
            class_name: Name of the strategy class (auto-detect if None)

        Returns:
            Strategy class
        """
        with open(file_path, 'r') as f:
            code = f.read()

        return StrategyLoader.load_from_code(code, class_name)

    @staticmethod
    def validate_strategy(strategy_class: Type) -> bool:
        """
        Validate that a strategy class has required methods

        Args:
            strategy_class: Strategy class to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required_methods = ['calculate_indicators', 'generate_signals']

        for method in required_methods:
            if not hasattr(strategy_class, method):
                raise ValueError(f"Strategy missing required method: {method}")

        # Try to instantiate
        try:
            instance = strategy_class()
        except Exception as e:
            raise ValueError(f"Failed to instantiate strategy: {e}")

        return True


def load_strategy_from_code(code: str, class_name: Optional[str] = None) -> Type:
    """
    Convenience function to load strategy from code

    Args:
        code: Python code as string
        class_name: Name of the strategy class (auto-detect if None)

    Returns:
        Strategy class
    """
    return StrategyLoader.load_from_code(code, class_name)


logger.info("Strategy loader module loaded")
