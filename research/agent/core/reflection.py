"""
Reflection and Learning

Analyzes experiment results and extracts insights for future research.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from loguru import logger

from research.agent.llm.interface import ILLM
from research.agent.memory.interface import Experiment, Principle, ISemanticMemory


@dataclass
class Insight:
    """An insight learned from experiments"""
    insight: str
    confidence: float  # 0-1
    supporting_experiments: List[str]  # Experiment IDs
    category: str  # "entry", "exit", "indicator", "regime", "risk_management"
    timestamp: datetime


class ReflectionEngine:
    """
    Analyzes results and learns from experiments

    Single Responsibility: Extract insights from experiments
    Open/Closed: Extend with new analysis methods
    """

    def __init__(
        self,
        llm: ILLM,
        semantic_memory: Optional[ISemanticMemory] = None
    ):
        self.llm = llm
        self.semantic_memory = semantic_memory
        logger.info("Initialized ReflectionEngine")

    async def analyze_experiment(
        self,
        experiment: Experiment,
        compare_to: Optional[List[Experiment]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single experiment and extract insights

        Args:
            experiment: The experiment to analyze
            compare_to: Optional similar experiments for comparison

        Returns:
            Analysis with insights, strengths, weaknesses
        """
        logger.info(f"Analyzing experiment: {experiment.id}")

        # Build analysis context
        context = self._build_analysis_context(experiment, compare_to)

        # Generate analysis using LLM
        system_prompt = self._get_analysis_system_prompt()
        user_prompt = self._build_analysis_prompt(context)

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2000,
            response_format="json"
        )

        # Parse response
        analysis = json.loads(response)

        # Add metadata
        analysis["experiment_id"] = experiment.id
        analysis["analyzed_at"] = datetime.now().isoformat()

        logger.info(f"Analysis complete. Success: {analysis.get('is_successful', False)}")
        return analysis

    async def extract_insights(
        self,
        experiments: List[Experiment],
        min_experiments: int = 3
    ) -> List[Insight]:
        """
        Extract general insights from multiple experiments

        Args:
            experiments: List of experiments to analyze
            min_experiments: Minimum experiments needed for insight

        Returns:
            List of insights learned
        """
        logger.info(f"Extracting insights from {len(experiments)} experiments")

        if len(experiments) < min_experiments:
            logger.warning(f"Need at least {min_experiments} experiments for insights")
            return []

        # Group experiments by outcome
        successful = [e for e in experiments if e.results.get("sharpe_ratio", 0) > 1.0]
        failed = [e for e in experiments if e.results.get("sharpe_ratio", 0) < 0.5]

        # Build context
        context = {
            "total_experiments": len(experiments),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful_patterns": self._extract_patterns(successful),
            "failed_patterns": self._extract_patterns(failed),
        }

        # Generate insights using LLM
        system_prompt = self._get_insight_system_prompt()
        user_prompt = self._build_insight_prompt(context, successful, failed)

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=2000,
            response_format="json"
        )

        # Parse insights
        insights_data = json.loads(response)
        insights = []

        for item in insights_data.get("insights", []):
            insight = Insight(
                insight=item["insight"],
                confidence=item["confidence"],
                supporting_experiments=item.get("supporting_experiments", []),
                category=item["category"],
                timestamp=datetime.now()
            )
            insights.append(insight)

        logger.info(f"Extracted {len(insights)} insights")

        # Save to semantic memory if available
        if self.semantic_memory:
            for insight in insights:
                principle = Principle(
                    principle=insight.insight,
                    category=insight.category,
                    confidence=insight.confidence,
                    supporting_evidence=insight.supporting_experiments
                )
                await self.semantic_memory.save_principle(principle)

        return insights

    async def suggest_next_experiments(
        self,
        past_experiments: List[Experiment],
        insights: List[Insight],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Suggest promising directions for next experiments

        Args:
            past_experiments: All past experiments
            insights: Learned insights
            count: Number of suggestions

        Returns:
            List of experiment suggestions
        """
        logger.info(f"Generating {count} experiment suggestions")

        # Build context
        context = {
            "past_count": len(past_experiments),
            "insights": [
                {
                    "insight": i.insight,
                    "confidence": i.confidence,
                    "category": i.category
                }
                for i in insights
            ],
            "best_results": sorted(
                past_experiments,
                key=lambda e: e.results.get("sharpe_ratio", 0),
                reverse=True
            )[:3],
            "unexplored_areas": self._identify_gaps(past_experiments)
        }

        system_prompt = self._get_suggestion_system_prompt()
        user_prompt = self._build_suggestion_prompt(context, count)

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2000,
            response_format="json"
        )

        suggestions = json.loads(response)
        logger.info(f"Generated {len(suggestions.get('suggestions', []))} suggestions")

        return suggestions.get("suggestions", [])

    async def compare_experiments(
        self,
        experiment_a: Experiment,
        experiment_b: Experiment
    ) -> Dict[str, Any]:
        """
        Compare two experiments and explain differences

        Args:
            experiment_a: First experiment
            experiment_b: Second experiment

        Returns:
            Comparison analysis
        """
        logger.info(f"Comparing {experiment_a.id} vs {experiment_b.id}")

        prompt = f"""Compare these two trading strategy experiments:

## Experiment A
Hypothesis: {experiment_a.hypothesis}
Results: {experiment_a.results}
Insights: {experiment_a.insights}

## Experiment B
Hypothesis: {experiment_b.hypothesis}
Results: {experiment_b.results}
Insights: {experiment_b.insights}

Analyze:
1. Key differences in approach
2. Why one performed better (if applicable)
3. What can be learned from the comparison

Return analysis in JSON:
{{
  "key_differences": ["difference1", "difference2"],
  "performance_analysis": "why one performed better",
  "learnings": ["learning1", "learning2"],
  "recommendation": "which approach to explore further"
}}
"""

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1500,
            response_format="json"
        )

        return json.loads(response)

    def _build_analysis_context(
        self,
        experiment: Experiment,
        compare_to: Optional[List[Experiment]]
    ) -> Dict[str, Any]:
        """Build context for experiment analysis"""
        context = {
            "experiment": {
                "hypothesis": experiment.hypothesis,
                "results": experiment.results,
                "code": experiment.code if hasattr(experiment, 'code') else None
            }
        }

        if compare_to:
            context["similar_experiments"] = [
                {
                    "hypothesis": e.hypothesis,
                    "results": e.results,
                    "insights": e.insights
                }
                for e in compare_to[:3]
            ]

        return context

    def _get_analysis_system_prompt(self) -> str:
        """System prompt for experiment analysis"""
        return """You are an expert quantitative analyst reviewing trading strategy experiments.

Your role is to:
1. Evaluate experiment results objectively
2. Identify what worked and what didn't
3. Extract actionable insights
4. Suggest improvements

Focus on:
- Statistical significance (Sharpe ratio, win rate, total trades)
- Strategy logic (entry/exit conditions, indicators)
- Risk management
- Market conditions and regime fit"""

    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build analysis prompt"""
        exp = context["experiment"]

        prompt = f"""Analyze this trading strategy experiment:

## Hypothesis
{json.dumps(exp['hypothesis'], indent=2)}

## Results
{json.dumps(exp['results'], indent=2)}

## Task
Provide comprehensive analysis in JSON format:

{{
  "is_successful": true/false,
  "success_criteria": "explain why successful or not",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "insights": [
    {{
      "category": "entry|exit|indicator|regime|risk_management",
      "insight": "specific insight learned",
      "confidence": 0.8
    }}
  ],
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "next_steps": ["what to try next based on these results"]
}}

Consider:
- Is Sharpe ratio > 1.0? (minimum for success)
- Is win rate reasonable? (45-65% typical)
- Are there enough trades? (30+ for statistical significance)
- Does the logic make sense given the results?
"""

        if context.get("similar_experiments"):
            prompt += f"\n## Similar Past Experiments\n"
            for i, sim in enumerate(context["similar_experiments"], 1):
                prompt += f"\n### Similar Experiment {i}\n"
                prompt += f"Results: {sim['results']}\n"
                prompt += f"Insights: {sim.get('insights', 'N/A')}\n"

        return prompt

    def _get_insight_system_prompt(self) -> str:
        """System prompt for insight extraction"""
        return """You are a quantitative researcher extracting general insights from experiment data.

Your role is to:
1. Find patterns across multiple experiments
2. Identify what generally works vs. what doesn't
3. Formulate actionable principles for future research

Be:
- Evidence-based (reference specific experiments)
- Specific (avoid vague advice)
- Confidence-calibrated (express uncertainty when appropriate)"""

    def _build_insight_prompt(
        self,
        context: Dict[str, Any],
        successful: List[Experiment],
        failed: List[Experiment]
    ) -> str:
        """Build insight extraction prompt"""
        prompt = f"""Extract general insights from these experiments:

## Summary
Total experiments: {context['total_experiments']}
Successful (Sharpe > 1.0): {context['successful_count']}
Failed (Sharpe < 0.5): {context['failed_count']}

## Successful Experiments
"""

        for exp in successful[:5]:
            prompt += f"\n- {exp.hypothesis.get('description', 'N/A')}\n"
            prompt += f"  Sharpe: {exp.results.get('sharpe_ratio', 'N/A')}, "
            prompt += f"Win Rate: {exp.results.get('win_rate', 'N/A')}\n"
            prompt += f"  Indicators: {exp.hypothesis.get('indicators', [])}\n"

        prompt += f"\n## Failed Experiments\n"

        for exp in failed[:5]:
            prompt += f"\n- {exp.hypothesis.get('description', 'N/A')}\n"
            prompt += f"  Sharpe: {exp.results.get('sharpe_ratio', 'N/A')}, "
            prompt += f"  Indicators: {exp.hypothesis.get('indicators', [])}\n"

        prompt += f"""

## Task
Extract general insights in JSON format:

{{
  "insights": [
    {{
      "insight": "Specific, actionable insight",
      "confidence": 0.7,
      "category": "entry|exit|indicator|regime|risk_management",
      "supporting_experiments": ["exp_id_1", "exp_id_2"]
    }}
  ]
}}

Focus on:
1. Which indicators work well together
2. Which market regimes favor which strategies
3. Common failure patterns to avoid
4. Risk management principles
5. Entry/exit timing patterns
"""

        return prompt

    def _get_suggestion_system_prompt(self) -> str:
        """System prompt for suggesting next experiments"""
        return """You are a quantitative research director planning the next experiments.

Your role is to:
1. Build on successful strategies
2. Explore promising unexplored areas
3. Avoid repeating failed approaches
4. Balance exploitation (refine winners) vs. exploration (try new ideas)

Suggest experiments that are:
- Informed by past learnings
- Diverse (explore different concepts)
- Testable via backtesting
- Specific and concrete"""

    def _build_suggestion_prompt(
        self,
        context: Dict[str, Any],
        count: int
    ) -> str:
        """Build suggestion prompt"""
        prompt = f"""Based on past experiments and insights, suggest {count} promising next experiments:

## Past Performance
Total experiments: {context['past_count']}

## Best Results So Far
"""

        for exp in context["best_results"]:
            prompt += f"\n- {exp.hypothesis.get('description', 'N/A')}\n"
            prompt += f"  Sharpe: {exp.results.get('sharpe_ratio', 'N/A')}\n"
            prompt += f"  Indicators: {exp.hypothesis.get('indicators', [])}\n"

        prompt += f"\n## Learned Insights\n"

        for insight in context["insights"][:5]:
            prompt += f"- [{insight['category']}] {insight['insight']} (confidence: {insight['confidence']})\n"

        prompt += f"\n## Unexplored Areas\n{context.get('unexplored_areas', 'None identified')}\n"

        prompt += f"""

## Task
Suggest {count} next experiments in JSON format:

{{
  "suggestions": [
    {{
      "focus": "Brief description of what to try",
      "rationale": "Why this is promising based on insights",
      "expected_outcome": "What we hope to learn",
      "priority": "high|medium|low"
    }}
  ]
}}

Balance:
- Exploitation: Refine successful approaches
- Exploration: Try new promising concepts
"""

        return prompt

    def _extract_patterns(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """Extract common patterns from experiments"""
        if not experiments:
            return {}

        # Count indicator usage
        indicator_counts = {}
        regime_counts = {}

        for exp in experiments:
            for indicator in exp.hypothesis.get("indicators", []):
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

            regime = exp.hypothesis.get("target_regime")
            if regime:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            "common_indicators": sorted(
                indicator_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "common_regimes": sorted(
                regime_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            "avg_sharpe": sum(e.results.get("sharpe_ratio", 0) for e in experiments) / len(experiments),
            "avg_win_rate": sum(e.results.get("win_rate", 0) for e in experiments) / len(experiments)
        }

    def _identify_gaps(self, experiments: List[Experiment]) -> str:
        """Identify unexplored areas"""
        tested_indicators = set()
        tested_regimes = set()

        for exp in experiments:
            tested_indicators.update(exp.hypothesis.get("indicators", []))
            regime = exp.hypothesis.get("target_regime")
            if regime:
                tested_regimes.add(regime)

        # Known indicators and regimes
        all_indicators = {
            "RSI", "MACD", "Bollinger Bands", "ATR", "ADX",
            "Stochastic", "OBV", "VWAP", "EMA", "SMA",
            "CCI", "Williams %R", "Ichimoku", "Fibonacci"
        }
        all_regimes = {"trending", "ranging", "volatile", "low_volatility"}

        unexplored_indicators = all_indicators - tested_indicators
        unexplored_regimes = all_regimes - tested_regimes

        gaps = []
        if unexplored_indicators:
            gaps.append(f"Indicators: {', '.join(list(unexplored_indicators)[:5])}")
        if unexplored_regimes:
            gaps.append(f"Regimes: {', '.join(unexplored_regimes)}")

        return "; ".join(gaps) if gaps else "Most areas explored"


logger.info("Reflection engine loaded")
