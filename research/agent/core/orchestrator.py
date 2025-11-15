"""
Research Orchestration

Coordinates the full research loop: hypothesis → code → backtest → reflect → iterate
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from loguru import logger

from research.agent.llm.interface import ILLM
from research.agent.core.hypothesis import HypothesisGenerator, StrategyHypothesis
from research.agent.core.code_generator import CodeGenerator, GeneratedCode
from research.agent.core.reflection import ReflectionEngine, Insight
from research.agent.memory.interface import (
    IEpisodicMemory,
    ISemanticMemory,
    IWorkingMemory,
    Experiment
)
from research.knowledge.integrated_retriever import IntegratedKnowledgeRetriever
from research.data.interface import IDataProvider, IFeatureEngineer, IPatternDetector


class ResearchPhase(Enum):
    """Phases of the research loop"""
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    CODE_GENERATION = "code_generation"
    BACKTESTING = "backtesting"
    REFLECTION = "reflection"
    LEARNING = "learning"
    COMPLETED = "completed"


@dataclass
class ResearchSession:
    """A research session"""
    id: str
    objective: str  # What we're trying to discover
    start_time: datetime
    end_time: Optional[datetime] = None
    experiments_run: int = 0
    successful_experiments: int = 0
    best_sharpe: float = 0.0
    best_strategy_id: Optional[str] = None
    insights_learned: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed


@dataclass
class ResearchConfig:
    """Configuration for research orchestrator"""
    max_iterations: int = 10
    min_sharpe_threshold: float = 1.5  # Minimum Sharpe to consider successful
    diversity_threshold: float = 0.7  # How different hypotheses should be
    parallel_experiments: int = 1  # How many to run in parallel
    backtest_symbol: str = "BTC/USDT"
    backtest_timeframe: str = "1h"
    backtest_start_date: str = "2024-01-01"
    backtest_end_date: str = "2024-06-01"
    learning_interval: int = 3  # Extract insights every N experiments


class ResearchOrchestrator:
    """
    Orchestrates the autonomous research loop

    Single Responsibility: Coordinate the research process
    Open/Closed: Extend with new research strategies
    """

    def __init__(
        self,
        llm: ILLM,
        knowledge_retriever: IntegratedKnowledgeRetriever,
        data_provider: IDataProvider,
        feature_engineer: IFeatureEngineer,
        pattern_detector: IPatternDetector,
        episodic_memory: IEpisodicMemory,
        semantic_memory: ISemanticMemory,
        working_memory: IWorkingMemory,
        config: Optional[ResearchConfig] = None
    ):
        self.llm = llm
        self.knowledge = knowledge_retriever
        self.data_provider = data_provider
        self.feature_engineer = feature_engineer
        self.pattern_detector = pattern_detector

        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory

        self.config = config or ResearchConfig()

        # Initialize components
        self.hypothesis_generator = HypothesisGenerator(llm, knowledge_retriever)
        self.code_generator = CodeGenerator(llm)
        self.reflection_engine = ReflectionEngine(llm, semantic_memory)

        self.current_session: Optional[ResearchSession] = None

        logger.info("Initialized ResearchOrchestrator")

    async def start_research_session(
        self,
        objective: str,
        focus: Optional[str] = None
    ) -> ResearchSession:
        """
        Start a new research session

        Args:
            objective: What we're trying to discover
            focus: Optional focus area (e.g., "mean reversion strategies")

        Returns:
            The research session
        """
        logger.info(f"Starting research session: {objective}")

        # Create session
        import hashlib
        session_id = hashlib.md5(
            f"{objective}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        self.current_session = ResearchSession(
            id=session_id,
            objective=objective,
            start_time=datetime.now()
        )

        # Store in working memory
        await self.working_memory.add_context(
            "research_objective",
            objective
        )
        if focus:
            await self.working_memory.add_context("research_focus", focus)

        logger.info(f"Research session started: {session_id}")
        return self.current_session

    async def run_research_loop(
        self,
        objective: str,
        focus: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> ResearchSession:
        """
        Run the full autonomous research loop

        This is the main entry point for autonomous research.

        Args:
            objective: Research objective
            focus: Optional focus area
            max_iterations: Override config max_iterations

        Returns:
            Completed research session
        """
        logger.info("=" * 80)
        logger.info(f"STARTING AUTONOMOUS RESEARCH: {objective}")
        logger.info("=" * 80)

        # Start session
        session = await self.start_research_session(objective, focus)

        max_iter = max_iterations or self.config.max_iterations
        insights: List[Insight] = []

        try:
            for iteration in range(1, max_iter + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"ITERATION {iteration}/{max_iter}")
                logger.info(f"{'='*80}\n")

                # Phase 1: Generate hypothesis
                logger.info("Phase 1: Generating hypothesis...")
                hypothesis = await self._generate_hypothesis(focus, insights)

                # Phase 2: Generate code
                logger.info("Phase 2: Generating code...")
                generated_code = await self._generate_code(hypothesis)

                if not generated_code.is_valid:
                    logger.error("Code generation failed, skipping iteration")
                    continue

                # Phase 3: Run backtest
                logger.info("Phase 3: Running backtest...")
                backtest_results = await self._run_backtest(generated_code)

                # Phase 4: Create experiment record
                experiment = await self._create_experiment(
                    hypothesis,
                    generated_code,
                    backtest_results
                )

                # Save to episodic memory
                await self.episodic_memory.save_experiment(experiment)

                # Update session stats
                session.experiments_run += 1
                sharpe = backtest_results.get("sharpe_ratio", 0)

                if sharpe > self.config.min_sharpe_threshold:
                    session.successful_experiments += 1

                if sharpe > session.best_sharpe:
                    session.best_sharpe = sharpe
                    session.best_strategy_id = experiment.id

                # Phase 5: Reflect on results
                logger.info("Phase 4: Reflecting on results...")
                analysis = await self.reflection_engine.analyze_experiment(
                    experiment
                )

                # Update experiment with insights
                experiment.insights = analysis.get("insights", [])
                await self.episodic_memory.save_experiment(experiment)

                # Phase 6: Extract general insights periodically
                if iteration % self.config.learning_interval == 0:
                    logger.info("Phase 5: Extracting general insights...")
                    all_experiments = await self.episodic_memory.get_all_experiments()
                    insights = await self.reflection_engine.extract_insights(
                        all_experiments
                    )

                    session.insights_learned = [i.insight for i in insights]

                # Log progress
                self._log_progress(session, iteration, max_iter, analysis)

                # Check if we found a good strategy
                if sharpe > self.config.min_sharpe_threshold * 1.5:
                    logger.success(f"Found excellent strategy! Sharpe: {sharpe:.2f}")

        except Exception as e:
            logger.error(f"Research loop failed: {e}")
            session.status = "failed"
            raise

        finally:
            # End session
            session.end_time = datetime.now()
            session.status = "completed"

            logger.info("\n" + "=" * 80)
            logger.info("RESEARCH SESSION COMPLETED")
            logger.info("=" * 80)
            self._log_session_summary(session)

        return session

    async def _generate_hypothesis(
        self,
        focus: Optional[str],
        insights: List[Insight]
    ) -> StrategyHypothesis:
        """Generate a new hypothesis"""
        # Get current market context
        try:
            market_data = await self.data_provider.fetch_ohlcv(
                symbol=self.config.backtest_symbol,
                timeframe=self.config.backtest_timeframe,
                start_date=datetime.fromisoformat(self.config.backtest_start_date),
                end_date=datetime.fromisoformat(self.config.backtest_end_date)
            )

            # Detect patterns
            patterns = await self.pattern_detector.detect_trends(market_data.data)
            regimes = await self.pattern_detector.detect_regimes(market_data.data)
            regime = regimes[0] if regimes else None

        except Exception as e:
            logger.warning(f"Failed to get market context: {e}")
            market_data = None
            patterns = None
            regime = None

        # Get concepts to avoid (from failed experiments)
        failed_experiments = await self.episodic_memory.query_experiments(
            filters={"max_sharpe": 0.5}
        )
        avoid_concepts = []
        for exp in failed_experiments[:5]:
            avoid_concepts.extend(exp.hypothesis.get("concepts_used", []))

        # Generate hypothesis
        hypothesis = await self.hypothesis_generator.generate_hypothesis(
            focus=focus,
            market_data=market_data,
            patterns=patterns,
            regime=regime,
            avoid_concepts=avoid_concepts
        )

        logger.info(f"Generated: {hypothesis.description}")
        return hypothesis

    async def _generate_code(
        self,
        hypothesis: StrategyHypothesis
    ) -> GeneratedCode:
        """Generate code from hypothesis"""
        generated = await self.code_generator.generate_strategy_code(hypothesis)

        if generated.is_valid:
            logger.success(f"Generated valid code: {generated.class_name}")
        else:
            logger.error(f"Code validation failed: {generated.validation_errors}")

        return generated

    async def _run_backtest(
        self,
        generated_code: GeneratedCode
    ) -> Dict[str, Any]:
        """
        Run backtest on generated strategy

        Note: This is a placeholder. In production, integrate with actual
        backtesting framework.
        """
        logger.info("Running backtest...")

        # TODO: Integrate with actual backtesting framework
        # For now, return mock results
        import random

        # Simulate backtest results
        await asyncio.sleep(0.5)  # Simulate backtest time

        results = {
            "sharpe_ratio": random.uniform(-0.5, 2.5),
            "total_return": random.uniform(-20, 50),
            "win_rate": random.uniform(0.35, 0.65),
            "total_trades": random.randint(20, 200),
            "max_drawdown": random.uniform(5, 30),
            "profit_factor": random.uniform(0.5, 2.5),
        }

        logger.info(f"Backtest complete. Sharpe: {results['sharpe_ratio']:.2f}")
        return results

    async def _create_experiment(
        self,
        hypothesis: StrategyHypothesis,
        code: GeneratedCode,
        results: Dict[str, Any]
    ) -> Experiment:
        """Create experiment record"""
        experiment = Experiment(
            id=hypothesis.id,
            hypothesis={
                "description": hypothesis.description,
                "rationale": hypothesis.rationale,
                "indicators": hypothesis.indicators,
                "entry_logic": hypothesis.entry_logic,
                "exit_logic": hypothesis.exit_logic,
                "concepts_used": hypothesis.concepts_used
            },
            code=code.code,
            results=results,
            timestamp=datetime.now(),
            insights=[],
            metadata={
                "class_name": code.class_name,
                "target_regime": hypothesis.target_regime,
                "expected_sharpe": hypothesis.expected_sharpe
            }
        )

        return experiment

    def _log_progress(
        self,
        session: ResearchSession,
        iteration: int,
        max_iterations: int,
        analysis: Dict[str, Any]
    ):
        """Log progress"""
        logger.info("\n" + "-" * 80)
        logger.info(f"Progress: {iteration}/{max_iterations}")
        logger.info(f"Experiments run: {session.experiments_run}")
        logger.info(f"Successful: {session.successful_experiments}")
        logger.info(f"Best Sharpe: {session.best_sharpe:.2f}")
        logger.info(f"Latest result: {'SUCCESS' if analysis.get('is_successful') else 'FAILED'}")
        logger.info("-" * 80 + "\n")

    def _log_session_summary(self, session: ResearchSession):
        """Log session summary"""
        duration = (session.end_time - session.start_time).total_seconds() / 60

        logger.info(f"\nObjective: {session.objective}")
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Experiments run: {session.experiments_run}")
        logger.info(f"Successful experiments: {session.successful_experiments}")
        logger.info(f"Success rate: {session.successful_experiments / max(session.experiments_run, 1) * 100:.1f}%")
        logger.info(f"Best Sharpe ratio: {session.best_sharpe:.2f}")
        logger.info(f"Best strategy ID: {session.best_strategy_id}")

        if session.insights_learned:
            logger.info(f"\nInsights learned ({len(session.insights_learned)}):")
            for i, insight in enumerate(session.insights_learned[:5], 1):
                logger.info(f"{i}. {insight}")

    async def get_best_strategies(
        self,
        top_k: int = 5,
        min_sharpe: float = 1.5
    ) -> List[Experiment]:
        """
        Get best strategies found so far

        Args:
            top_k: Number of strategies to return
            min_sharpe: Minimum Sharpe ratio

        Returns:
            List of best experiments
        """
        experiments = await self.episodic_memory.get_successful_experiments(
            min_sharpe=min_sharpe,
            min_trades=30
        )

        # Sort by Sharpe ratio
        experiments.sort(
            key=lambda e: e.results.get("sharpe_ratio", 0),
            reverse=True
        )

        return experiments[:top_k]

    async def generate_research_report(
        self,
        session: Optional[ResearchSession] = None
    ) -> str:
        """
        Generate a comprehensive research report

        Args:
            session: Optional specific session (defaults to current)

        Returns:
            Markdown formatted report
        """
        if session is None:
            session = self.current_session

        if session is None:
            return "No research session available"

        # Get all experiments from session
        all_experiments = await self.episodic_memory.get_all_experiments()

        # Get best strategies
        best_strategies = await self.get_best_strategies(top_k=5)

        # Get insights
        all_insights = await self.semantic_memory.get_all_principles()

        # Build report
        report = f"""# Research Report: {session.objective}

## Session Summary
- **Session ID**: {session.id}
- **Duration**: {(session.end_time - session.start_time).total_seconds() / 60:.1f} minutes
- **Experiments Run**: {session.experiments_run}
- **Successful Experiments**: {session.successful_experiments} ({session.successful_experiments / max(session.experiments_run, 1) * 100:.1f}%)
- **Best Sharpe Ratio**: {session.best_sharpe:.2f}

## Top Strategies

"""

        for i, exp in enumerate(best_strategies, 1):
            report += f"### {i}. {exp.hypothesis.get('description', 'N/A')}\n\n"
            report += f"**Performance**:\n"
            report += f"- Sharpe Ratio: {exp.results.get('sharpe_ratio', 'N/A'):.2f}\n"
            report += f"- Total Return: {exp.results.get('total_return', 'N/A'):.1f}%\n"
            report += f"- Win Rate: {exp.results.get('win_rate', 'N/A'):.1f}%\n"
            report += f"- Total Trades: {exp.results.get('total_trades', 'N/A')}\n\n"

            report += f"**Strategy**:\n"
            report += f"- Indicators: {', '.join(exp.hypothesis.get('indicators', []))}\n"
            report += f"- Entry: {exp.hypothesis.get('entry_logic', 'N/A')}\n"
            report += f"- Exit: {exp.hypothesis.get('exit_logic', 'N/A')}\n\n"

        report += f"\n## Insights Learned\n\n"

        for i, principle in enumerate(all_insights[:10], 1):
            report += f"{i}. **[{principle.category}]** {principle.principle}\n"
            report += f"   - Confidence: {principle.confidence:.2f}\n\n"

        report += f"\n## Next Steps\n\n"

        # Get suggestions
        insights_list = [
            Insight(
                insight=p.principle,
                confidence=p.confidence,
                supporting_experiments=p.supporting_evidence,
                category=p.category,
                timestamp=datetime.now()
            )
            for p in all_insights
        ]

        suggestions = await self.reflection_engine.suggest_next_experiments(
            all_experiments,
            insights_list,
            count=3
        )

        for i, sug in enumerate(suggestions, 1):
            report += f"{i}. {sug.get('focus', 'N/A')}\n"
            report += f"   - Rationale: {sug.get('rationale', 'N/A')}\n"
            report += f"   - Priority: {sug.get('priority', 'medium')}\n\n"

        return report


logger.info("Research orchestrator loaded")
