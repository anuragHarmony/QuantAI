"""
Code Generation

Converts strategy hypotheses into executable Python code.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
import ast
import re
from loguru import logger

from research.agent.llm.interface import ILLM
from research.agent.core.hypothesis import StrategyHypothesis


@dataclass
class GeneratedCode:
    """Generated strategy code"""
    code: str
    class_name: str
    hypothesis_id: str
    is_valid: bool
    validation_errors: list[str]
    metadata: Dict[str, Any]


class CodeGenerator:
    """
    Generates executable Python code from strategy hypotheses

    Single Responsibility: Convert hypothesis → Python code
    Open/Closed: Extend with new code templates or validation rules
    """

    def __init__(self, llm: ILLM):
        self.llm = llm
        logger.info("Initialized CodeGenerator")

    async def generate_strategy_code(
        self,
        hypothesis: StrategyHypothesis,
        base_class: str = "BaseStrategy"
    ) -> GeneratedCode:
        """
        Generate executable strategy code from hypothesis

        Args:
            hypothesis: The strategy hypothesis to implement
            base_class: Base class to inherit from

        Returns:
            Generated code with validation results
        """
        logger.info(f"Generating code for: {hypothesis.description}")

        # Step 1: Generate code using LLM
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_code_prompt(hypothesis, base_class)

        code = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for precise code
            max_tokens=3000
        )

        # Step 2: Extract code from response
        code = self._extract_code(code)

        # Step 3: Extract class name
        class_name = self._extract_class_name(code)

        # Step 4: Validate code
        is_valid, errors = self._validate_code(code)

        # Step 5: If invalid, try to fix once
        if not is_valid and len(errors) <= 3:
            logger.warning(f"Code validation failed, attempting fix: {errors}")
            code = await self._fix_code(code, errors)
            is_valid, errors = self._validate_code(code)

        generated = GeneratedCode(
            code=code,
            class_name=class_name,
            hypothesis_id=hypothesis.id,
            is_valid=is_valid,
            validation_errors=errors,
            metadata={
                "indicators": hypothesis.indicators,
                "entry_logic": hypothesis.entry_logic,
                "exit_logic": hypothesis.exit_logic
            }
        )

        if is_valid:
            logger.info(f"Successfully generated valid code: {class_name}")
        else:
            logger.error(f"Code validation failed: {errors}")

        return generated

    def _get_system_prompt(self) -> str:
        """System prompt for code generation"""
        return """You are an expert Python developer specializing in algorithmic trading systems.

Your role is to convert trading strategy hypotheses into clean, executable Python code.

Requirements:
- Inherit from BaseStrategy class
- Implement calculate_indicators() method
- Implement generate_signals() method
- Use pandas DataFrames for data manipulation
- Handle edge cases (insufficient data, NaN values)
- Follow PEP 8 style guidelines
- Add clear docstrings
- Use type hints

The code must be:
- Executable without errors
- Efficient and vectorized (avoid loops where possible)
- Well-documented
- Ready for backtesting"""

    def _build_code_prompt(
        self,
        hypothesis: StrategyHypothesis,
        base_class: str
    ) -> str:
        """Build prompt for code generation"""
        # Create sanitized class name
        class_name = self._sanitize_class_name(hypothesis.description)

        prompt = f"""Generate a complete Python trading strategy class based on this hypothesis:

## Hypothesis
Description: {hypothesis.description}
Rationale: {hypothesis.rationale}

## Strategy Details
Indicators: {', '.join(hypothesis.indicators)}
Entry Logic: {hypothesis.entry_logic}
Exit Logic: {hypothesis.exit_logic}
Risk Management: {hypothesis.risk_management}
Target Regime: {hypothesis.target_regime or 'any'}

## Base Class Template

```python
class BaseStrategy:
    \"\"\"Base class for trading strategies\"\"\"

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {{}}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Calculate indicators needed for the strategy\"\"\"
        raise NotImplementedError

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Generate trading signals (1=buy, -1=sell, 0=hold)\"\"\"
        raise NotImplementedError
```

## Task

Generate a complete implementation inheriting from {base_class}:

1. Class name: {class_name}
2. Implement calculate_indicators() to calculate: {', '.join(hypothesis.indicators)}
3. Implement generate_signals() with:
   - Entry logic: {hypothesis.entry_logic}
   - Exit logic: {hypothesis.exit_logic}
4. Add a docstring explaining the strategy
5. Use the risk management parameters: {hypothesis.risk_management}

## Example Structure

```python
from typing import Dict, Any
import pandas as pd
import numpy as np

class {class_name}(BaseStrategy):
    \"\"\"
    {hypothesis.description}

    Strategy: {hypothesis.rationale}

    Entry: {hypothesis.entry_logic}
    Exit: {hypothesis.exit_logic}
    \"\"\"

    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__(name=\"{class_name}\", parameters=parameters)

        # Set default parameters from hypothesis
        self.parameters.setdefault('stop_loss_pct', {hypothesis.risk_management.get('stop_loss_pct', 2.0)})
        self.parameters.setdefault('position_size_pct', {hypothesis.risk_management.get('position_size_pct', 10.0)})
        # Add indicator parameters here

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Calculate indicators\"\"\"
        df = data.copy()

        # Implement indicator calculations here
        # Example: df['rsi_14'] = self._calculate_rsi(df['close'], 14)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Generate trading signals\"\"\"
        df = data.copy()

        # Calculate indicators first
        df = self.calculate_indicators(df)

        # Initialize signal column
        df['signal'] = 0

        # Implement entry logic
        # entry_condition = (df['rsi_14'] < 30) & ...
        # df.loc[entry_condition, 'signal'] = 1

        # Implement exit logic
        # exit_condition = (df['rsi_14'] > 70) | ...
        # df.loc[exit_condition, 'signal'] = -1

        return df
```

Generate the complete, executable code. Return ONLY the Python code, no explanations."""

        return prompt

    def _extract_code(self, llm_response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to extract code from markdown code blocks
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code block, try to extract class definition
        pattern = r"(class\s+\w+.*)"
        matches = re.findall(pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Return as-is if no patterns match
        return llm_response.strip()

    def _extract_class_name(self, code: str) -> str:
        """Extract class name from code"""
        pattern = r"class\s+(\w+)"
        match = re.search(pattern, code)

        if match:
            return match.group(1)

        return "UnknownStrategy"

    def _sanitize_class_name(self, description: str) -> str:
        """Convert description to valid Python class name"""
        # Remove special characters, keep only alphanumeric and spaces
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', description)

        # Convert to title case and remove spaces
        words = clean.split()
        class_name = ''.join(word.capitalize() for word in words[:5])  # Max 5 words

        # Ensure it starts with letter
        if not class_name or not class_name[0].isalpha():
            class_name = 'Strategy' + class_name

        return class_name + 'Strategy'

    def _validate_code(self, code: str) -> tuple[bool, list[str]]:
        """
        Validate generated code

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check 1: Valid Python syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors

        # Check 2: Contains class definition
        if not re.search(r"class\s+\w+", code):
            errors.append("No class definition found")

        # Check 3: Contains required methods
        if "calculate_indicators" not in code:
            errors.append("Missing calculate_indicators method")

        if "generate_signals" not in code:
            errors.append("Missing generate_signals method")

        # Check 4: Imports necessary modules
        has_pandas = "import pandas" in code or "from pandas" in code
        has_numpy = "import numpy" in code or "from numpy" in code

        if not has_pandas:
            errors.append("Missing pandas import")

        # Check 5: Returns DataFrame (basic check)
        if "return df" not in code and "return data" not in code:
            errors.append("Methods should return DataFrames")

        is_valid = len(errors) == 0
        return is_valid, errors

    async def _fix_code(self, code: str, errors: list[str]) -> str:
        """Attempt to fix code validation errors"""
        logger.info(f"Attempting to fix {len(errors)} errors")

        prompt = f"""This generated code has errors. Please fix them:

## Errors
{chr(10).join(f'- {error}' for error in errors)}

## Code
```python
{code}
```

Return the corrected code. Return ONLY the Python code, no explanations."""

        fixed_code = await self.llm.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=3000
        )

        return self._extract_code(fixed_code)

    async def save_to_file(
        self,
        generated: GeneratedCode,
        file_path: str
    ) -> bool:
        """
        Save generated code to file

        Args:
            generated: Generated code object
            file_path: Path to save the code

        Returns:
            True if saved successfully
        """
        if not generated.is_valid:
            logger.error(f"Cannot save invalid code: {generated.validation_errors}")
            return False

        try:
            # Add header comment
            header = f'''"""
Auto-generated strategy code
Hypothesis ID: {generated.hypothesis_id}
Generated: {pd.Timestamp.now()}

{generated.metadata.get('entry_logic', '')}
"""

'''
            full_code = header + generated.code

            with open(file_path, 'w') as f:
                f.write(full_code)

            logger.info(f"Saved code to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save code: {e}")
            return False


logger.info("Code generator loaded")
