"""
Hypothesis Generation

Generates strategy hypotheses using LLM + knowledge sources.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from loguru import logger

from research.agent.llm.interface import ILLM
from research.knowledge.integrated_retriever import IntegratedKnowledgeRetriever
from research.data.interface import MarketData, Pattern, MarketRegime


@dataclass
class Hypothesis:
    """A research hypothesis"""
    id: str
    description: str
    rationale: str
    concepts_used: List[str]  # Concept IDs from knowledge graph
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyHypothesis(Hypothesis):
    """
    A trading strategy hypothesis

    Single Responsibility: Represent a testable strategy idea
    """
    # Core strategy elements
    indicators: List[str] = field(default_factory=list)  # e.g., ["RSI", "MACD", "Bollinger Bands"]
    entry_logic: str = ""  # Description of entry conditions
    exit_logic: str = ""  # Description of exit conditions
    risk_management: Dict[str, Any] = field(default_factory=dict)  # Stop loss, position sizing, etc.

    # Market context
    target_regime: Optional[str] = None  # "trending", "ranging", etc.
    target_symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1h"])

    # Expectations
    expected_sharpe: Optional[float] = None
    expected_win_rate: Optional[float] = None

    # Related past experiments
    similar_experiments: List[str] = field(default_factory=list)


class HypothesisGenerator:
    """
    Generates strategy hypotheses using LLM + knowledge sources

    Single Responsibility: Create testable strategy ideas
    Open/Closed: Can extend with new generation strategies
    """

    def __init__(
        self,
        llm: ILLM,
        knowledge_retriever: IntegratedKnowledgeRetriever
    ):
        self.llm = llm
        self.knowledge = knowledge_retriever
        logger.info("Initialized HypothesisGenerator")

    async def generate_hypothesis(
        self,
        focus: Optional[str] = None,
        market_data: Optional[MarketData] = None,
        patterns: Optional[List[Pattern]] = None,
        regime: Optional[MarketRegime] = None,
        avoid_concepts: Optional[List[str]] = None,
    ) -> StrategyHypothesis:
        """
        Generate a new strategy hypothesis

        Args:
            focus: Optional focus area (e.g., "mean reversion", "momentum")
            market_data: Current market data for context
            patterns: Detected patterns in current data
            regime: Current market regime
            avoid_concepts: Concepts to avoid (already tested)

        Returns:
            A new strategy hypothesis to test
        """
        logger.info(f"Generating hypothesis with focus: {focus}")

        # Step 1: Query knowledge sources
        query = focus if focus else "profitable trading strategies"
        knowledge = await self.knowledge.query(
            query=query,
            include_rag=True,
            include_graph=True,
            include_memory=True,
            context={
                "regime": regime.regime_type if regime else None,
                "avoid_concepts": avoid_concepts or []
            }
        )

        # Step 2: Build context for LLM
        context = self._build_context(
            knowledge=knowledge,
            market_data=market_data,
            patterns=patterns,
            regime=regime,
            avoid_concepts=avoid_concepts
        )

        # Step 3: Generate hypothesis using LLM
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_hypothesis_prompt(context, focus)

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.8,  # Higher temperature for creativity
            max_tokens=2000,
            response_format="json"
        )

        # Step 4: Parse LLM response into StrategyHypothesis
        hypothesis = self._parse_hypothesis(response, knowledge)

        logger.info(f"Generated hypothesis: {hypothesis.description}")
        return hypothesis

    async def generate_variation(
        self,
        base_hypothesis: StrategyHypothesis,
        variation_type: str = "improve"
    ) -> StrategyHypothesis:
        """
        Generate a variation of an existing hypothesis

        Args:
            base_hypothesis: Hypothesis to vary
            variation_type: "improve", "simplify", "different_regime", etc.

        Returns:
            A new hypothesis based on the original
        """
        logger.info(f"Generating {variation_type} variation")

        # Query for related concepts
        knowledge = await self.knowledge.query(
            query=base_hypothesis.description,
            include_graph=True,
            include_memory=True
        )

        system_prompt = self._get_system_prompt()
        user_prompt = self._build_variation_prompt(
            base_hypothesis,
            variation_type,
            knowledge
        )

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
            response_format="json"
        )

        hypothesis = self._parse_hypothesis(response, knowledge)
        hypothesis.metadata["variation_of"] = base_hypothesis.id
        hypothesis.metadata["variation_type"] = variation_type

        return hypothesis

    async def generate_batch(
        self,
        count: int = 3,
        diversity: float = 0.8,
        **kwargs
    ) -> List[StrategyHypothesis]:
        """
        Generate multiple diverse hypotheses

        Args:
            count: Number of hypotheses to generate
            diversity: How different they should be (0-1)
            **kwargs: Arguments passed to generate_hypothesis

        Returns:
            List of diverse hypotheses
        """
        logger.info(f"Generating batch of {count} hypotheses")

        hypotheses = []
        avoid_concepts = kwargs.get("avoid_concepts", [])

        for i in range(count):
            # After each generation, avoid similar concepts for diversity
            if i > 0 and diversity > 0.5:
                # Add concepts from previous hypotheses to avoid list
                for h in hypotheses:
                    avoid_concepts.extend(h.concepts_used)

            hypothesis = await self.generate_hypothesis(
                avoid_concepts=avoid_concepts,
                **kwargs
            )

            hypotheses.append(hypothesis)

            # Brief pause between generations for diversity
            import asyncio
            await asyncio.sleep(0.1)

        logger.info(f"Generated {len(hypotheses)} diverse hypotheses")
        return hypotheses

    def _build_context(
        self,
        knowledge: Any,
        market_data: Optional[MarketData],
        patterns: Optional[List[Pattern]],
        regime: Optional[MarketRegime],
        avoid_concepts: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build context dictionary for LLM"""
        context = {
            "knowledge": {
                "concepts": [
                    {
                        "name": c.name,
                        "type": c.type.value,
                        "description": c.description
                    }
                    for c in knowledge.graph_concepts[:5]  # Top 5 relevant concepts
                ],
                "past_experiments": [
                    {
                        "hypothesis": exp.hypothesis,
                        "results": exp.results,
                        "insights": exp.insights
                    }
                    for exp in knowledge.memory_experiments[:3]  # Top 3 similar experiments
                ],
                "textual_knowledge": knowledge.rag_results[:500] if knowledge.rag_results else None
            }
        }

        if market_data:
            context["market"] = {
                "symbol": market_data.symbol,
                "current_price": market_data.data["close"].iloc[-1] if not market_data.data.empty else None,
                "data_points": len(market_data.data)
            }

        if patterns:
            context["patterns"] = [
                {
                    "type": p.pattern_type,
                    "confidence": p.confidence,
                    "properties": p.properties
                }
                for p in patterns[:3]
            ]

        if regime:
            context["regime"] = {
                "type": regime.regime_type,
                "confidence": regime.confidence,
                "characteristics": regime.characteristics
            }

        if avoid_concepts:
            context["avoid"] = avoid_concepts

        return context

    def _get_system_prompt(self) -> str:
        """Get system prompt for hypothesis generation"""
        return """You are an expert quantitative researcher specializing in algorithmic trading strategies.

Your role is to generate creative, testable trading strategy hypotheses based on:
1. Quantitative finance concepts and indicators
2. Market data and patterns
3. Past experimental results

Key principles:
- Be specific and concrete about entry/exit logic
- Use well-defined technical indicators
- Consider risk management
- Match strategies to market regimes
- Learn from past successes and failures
- Balance novelty with practicality

Generate strategies that are:
- Testable via backtesting
- Based on sound quantitative principles
- Specific enough to code
- Different from failed past attempts"""

    def _build_hypothesis_prompt(
        self,
        context: Dict[str, Any],
        focus: Optional[str]
    ) -> str:
        """Build the hypothesis generation prompt"""
        prompt = "Generate a new trading strategy hypothesis.\n\n"

        if focus:
            prompt += f"Focus area: {focus}\n\n"

        prompt += "## Available Knowledge\n\n"

        # Add concepts
        if context.get("knowledge", {}).get("concepts"):
            prompt += "### Relevant Concepts:\n"
            for concept in context["knowledge"]["concepts"]:
                prompt += f"- {concept['name']} ({concept['type']}): {concept['description']}\n"
            prompt += "\n"

        # Add past experiments
        if context.get("knowledge", {}).get("past_experiments"):
            prompt += "### Past Experiments (learn from these):\n"
            for exp in context["knowledge"]["past_experiments"]:
                sharpe = exp["results"].get("sharpe_ratio", "N/A")
                prompt += f"- {exp['hypothesis'].get('description', 'N/A')}\n"
                prompt += f"  Sharpe: {sharpe}, Insights: {exp.get('insights', 'N/A')}\n"
            prompt += "\n"

        # Add market context
        if context.get("regime"):
            prompt += f"### Current Market Regime: {context['regime']['type']}\n"
            prompt += f"Characteristics: {context['regime']['characteristics']}\n\n"

        if context.get("patterns"):
            prompt += "### Detected Patterns:\n"
            for pattern in context["patterns"]:
                prompt += f"- {pattern['type']} (confidence: {pattern['confidence']})\n"
            prompt += "\n"

        # Add constraints
        if context.get("avoid"):
            prompt += f"### Concepts to Avoid (already tested): {', '.join(context['avoid'])}\n\n"

        prompt += """## Task

Generate a complete strategy hypothesis in JSON format with these fields:

{
  "description": "Brief description of the strategy (1-2 sentences)",
  "rationale": "Why this strategy might work (reference concepts, market conditions, or past learnings)",
  "indicators": ["List", "of", "technical", "indicators", "needed"],
  "entry_logic": "Detailed entry conditions (be specific with thresholds)",
  "exit_logic": "Detailed exit conditions (profit target, stop loss, time-based)",
  "risk_management": {
    "stop_loss_pct": 2.0,
    "position_size_pct": 10.0,
    "max_positions": 1
  },
  "target_regime": "trending|ranging|volatile|any",
  "expected_sharpe": 1.5,
  "expected_win_rate": 0.55
}

Be creative but practical. The strategy must be backtestable."""

        return prompt

    def _build_variation_prompt(
        self,
        base_hypothesis: StrategyHypothesis,
        variation_type: str,
        knowledge: Any
    ) -> str:
        """Build prompt for generating a variation"""
        prompt = f"Generate a {variation_type} variation of this strategy:\n\n"

        prompt += f"## Base Strategy\n"
        prompt += f"Description: {base_hypothesis.description}\n"
        prompt += f"Rationale: {base_hypothesis.rationale}\n"
        prompt += f"Indicators: {', '.join(base_hypothesis.indicators)}\n"
        prompt += f"Entry: {base_hypothesis.entry_logic}\n"
        prompt += f"Exit: {base_hypothesis.exit_logic}\n"
        prompt += f"Risk Management: {base_hypothesis.risk_management}\n\n"

        if base_hypothesis.metadata.get("backtest_results"):
            prompt += f"## Past Results\n"
            prompt += f"{base_hypothesis.metadata['backtest_results']}\n\n"

        prompt += f"## Variation Type: {variation_type}\n\n"

        if variation_type == "improve":
            prompt += "Improve the strategy by:\n"
            prompt += "- Refining entry/exit conditions\n"
            prompt += "- Adding filters to reduce false signals\n"
            prompt += "- Optimizing risk management\n"
        elif variation_type == "simplify":
            prompt += "Simplify the strategy by:\n"
            prompt += "- Using fewer indicators\n"
            prompt += "- Clearer entry/exit rules\n"
            prompt += "- Removing complexity\n"
        elif variation_type == "different_regime":
            prompt += "Adapt the strategy for a different market regime\n"

        prompt += "\nReturn the variation in the same JSON format as before."
        return prompt

    def _parse_hypothesis(
        self,
        llm_response: str,
        knowledge: Any
    ) -> StrategyHypothesis:
        """Parse LLM response into StrategyHypothesis"""
        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {llm_response}")
            raise

        # Extract concepts used from knowledge
        concepts_used = [c.id for c in knowledge.graph_concepts[:5]]

        # Generate ID
        import hashlib
        hypothesis_id = hashlib.md5(
            f"{data['description']}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Extract similar experiments
        similar_experiments = [exp.id for exp in knowledge.memory_experiments[:3]]

        hypothesis = StrategyHypothesis(
            id=hypothesis_id,
            description=data["description"],
            rationale=data["rationale"],
            concepts_used=concepts_used,
            indicators=data["indicators"],
            entry_logic=data["entry_logic"],
            exit_logic=data["exit_logic"],
            risk_management=data.get("risk_management", {}),
            target_regime=data.get("target_regime"),
            target_symbols=data.get("target_symbols", ["BTC/USDT"]),
            timeframes=data.get("timeframes", ["1h"]),
            expected_sharpe=data.get("expected_sharpe"),
            expected_win_rate=data.get("expected_win_rate"),
            similar_experiments=similar_experiments
        )

        return hypothesis


logger.info("Hypothesis generator loaded")
