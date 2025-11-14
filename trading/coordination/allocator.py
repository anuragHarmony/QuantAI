"""
Capital Allocator

Manages capital allocation across multiple strategies with different algorithms:
- Fixed allocation: Pre-defined percentages
- Dynamic allocation: Adjust based on recent performance
- Performance-based: Allocate more to better performers
- Risk-parity: Equal risk contribution from each strategy
"""
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger


@dataclass
class AllocationResult:
    """Result of capital allocation"""
    allocations: Dict[str, Decimal]  # strategy_id -> allocation percentage
    total_allocated: Decimal
    rebalance_required: bool = False


class CapitalAllocator(ABC):
    """Base class for capital allocation strategies"""

    def __init__(self, total_capital: Decimal):
        """
        Initialize allocator

        Args:
            total_capital: Total capital to allocate
        """
        self.total_capital = total_capital

    @abstractmethod
    def calculate_allocations(
        self,
        strategy_ids: List[str],
        performance_metrics: Optional[Dict[str, Dict[str, Decimal]]] = None
    ) -> AllocationResult:
        """
        Calculate capital allocations

        Args:
            strategy_ids: List of strategy IDs
            performance_metrics: Optional performance metrics per strategy

        Returns:
            AllocationResult with allocations
        """
        pass

    def validate_allocations(self, allocations: Dict[str, Decimal]) -> bool:
        """Validate that allocations sum to 1.0"""

        total = sum(allocations.values())

        # Allow small floating point error
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            logger.warning(f"Allocations sum to {total}, not 1.0")
            return False

        return True


class FixedAllocation(CapitalAllocator):
    """
    Fixed allocation strategy

    Allocates capital based on pre-defined percentages.
    """

    def __init__(
        self,
        total_capital: Decimal,
        allocations: Dict[str, Decimal]
    ):
        """
        Initialize fixed allocator

        Args:
            total_capital: Total capital
            allocations: Dict of strategy_id -> allocation percentage
        """
        super().__init__(total_capital)
        self.fixed_allocations = allocations

        if not self.validate_allocations(allocations):
            raise ValueError("Allocations must sum to 1.0")

        logger.info(f"Fixed allocator initialized with {len(allocations)} strategies")

    def calculate_allocations(
        self,
        strategy_ids: List[str],
        performance_metrics: Optional[Dict[str, Dict[str, Decimal]]] = None
    ) -> AllocationResult:
        """Return fixed allocations"""

        # Use fixed allocations for known strategies
        allocations = {}
        for strategy_id in strategy_ids:
            if strategy_id in self.fixed_allocations:
                allocations[strategy_id] = self.fixed_allocations[strategy_id]
            else:
                logger.warning(f"No fixed allocation for {strategy_id}, using 0")
                allocations[strategy_id] = Decimal("0")

        total = sum(allocations.values())

        return AllocationResult(
            allocations=allocations,
            total_allocated=total,
            rebalance_required=False
        )


class DynamicAllocation(CapitalAllocator):
    """
    Dynamic allocation strategy

    Adjusts allocations based on recent performance, with constraints.
    """

    def __init__(
        self,
        total_capital: Decimal,
        lookback_periods: int = 30,
        min_allocation: Decimal = Decimal("0.05"),  # Min 5%
        max_allocation: Decimal = Decimal("0.50"),  # Max 50%
        rebalance_threshold: Decimal = Decimal("0.10")  # Rebalance if drift > 10%
    ):
        """
        Initialize dynamic allocator

        Args:
            total_capital: Total capital
            lookback_periods: How many periods to look back for performance
            min_allocation: Minimum allocation per strategy
            max_allocation: Maximum allocation per strategy
            rebalance_threshold: Trigger rebalance if allocation drifts by this amount
        """
        super().__init__(total_capital)
        self.lookback_periods = lookback_periods
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.rebalance_threshold = rebalance_threshold

        # Track current allocations
        self.current_allocations: Dict[str, Decimal] = {}

        logger.info(
            f"Dynamic allocator initialized: "
            f"Range: {min_allocation * 100:.0f}%-{max_allocation * 100:.0f}%"
        )

    def calculate_allocations(
        self,
        strategy_ids: List[str],
        performance_metrics: Optional[Dict[str, Dict[str, Decimal]]] = None
    ) -> AllocationResult:
        """Calculate dynamic allocations based on performance"""

        if not performance_metrics:
            # No performance data - use equal allocation
            return self._equal_allocation(strategy_ids)

        # Calculate scores based on Sharpe ratio (or other metric)
        scores = {}
        for strategy_id in strategy_ids:
            metrics = performance_metrics.get(strategy_id, {})

            # Use Sharpe ratio as primary score
            sharpe = metrics.get("sharpe_ratio", Decimal("0"))

            # Ensure non-negative scores
            scores[strategy_id] = max(sharpe, Decimal("0"))

        # If all scores are zero, use equal allocation
        total_score = sum(scores.values())
        if total_score == 0:
            return self._equal_allocation(strategy_ids)

        # Calculate proportional allocations
        raw_allocations = {}
        for strategy_id, score in scores.items():
            raw_allocations[strategy_id] = score / total_score

        # Apply constraints
        allocations = self._apply_constraints(raw_allocations)

        # Normalize to ensure sum = 1.0
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}

        # Check if rebalance is needed
        rebalance_needed = self._should_rebalance(allocations)

        # Update current allocations
        if rebalance_needed:
            self.current_allocations = allocations.copy()

        return AllocationResult(
            allocations=allocations,
            total_allocated=sum(allocations.values()),
            rebalance_required=rebalance_needed
        )

    def _apply_constraints(self, allocations: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """Apply min/max constraints"""

        constrained = {}

        for strategy_id, allocation in allocations.items():
            # Apply min/max
            constrained[strategy_id] = max(
                self.min_allocation,
                min(self.max_allocation, allocation)
            )

        return constrained

    def _should_rebalance(self, new_allocations: Dict[str, Decimal]) -> bool:
        """Check if rebalancing is needed"""

        if not self.current_allocations:
            return True  # First time

        # Check maximum drift
        max_drift = Decimal("0")
        for strategy_id, new_alloc in new_allocations.items():
            old_alloc = self.current_allocations.get(strategy_id, Decimal("0"))
            drift = abs(new_alloc - old_alloc)
            max_drift = max(max_drift, drift)

        return max_drift > self.rebalance_threshold

    def _equal_allocation(self, strategy_ids: List[str]) -> AllocationResult:
        """Equal allocation fallback"""

        if not strategy_ids:
            return AllocationResult(allocations={}, total_allocated=Decimal("0"))

        equal_share = Decimal("1.0") / Decimal(str(len(strategy_ids)))
        allocations = {strategy_id: equal_share for strategy_id in strategy_ids}

        return AllocationResult(
            allocations=allocations,
            total_allocated=Decimal("1.0"),
            rebalance_required=False
        )


class PerformanceBasedAllocation(CapitalAllocator):
    """
    Performance-based allocation

    Allocates more capital to strategies with better risk-adjusted returns.
    Uses a combination of Sharpe ratio, win rate, and recent performance.
    """

    def __init__(
        self,
        total_capital: Decimal,
        sharpe_weight: Decimal = Decimal("0.5"),
        win_rate_weight: Decimal = Decimal("0.3"),
        recent_pnl_weight: Decimal = Decimal("0.2"),
        min_allocation: Decimal = Decimal("0.05"),
        max_allocation: Decimal = Decimal("0.60")
    ):
        """
        Initialize performance-based allocator

        Args:
            total_capital: Total capital
            sharpe_weight: Weight for Sharpe ratio
            win_rate_weight: Weight for win rate
            recent_pnl_weight: Weight for recent P&L
            min_allocation: Minimum allocation
            max_allocation: Maximum allocation
        """
        super().__init__(total_capital)
        self.sharpe_weight = sharpe_weight
        self.win_rate_weight = win_rate_weight
        self.recent_pnl_weight = recent_pnl_weight
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        # Weights should sum to 1.0
        total_weight = sharpe_weight + win_rate_weight + recent_pnl_weight
        if abs(total_weight - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        logger.info("Performance-based allocator initialized")

    def calculate_allocations(
        self,
        strategy_ids: List[str],
        performance_metrics: Optional[Dict[str, Dict[str, Decimal]]] = None
    ) -> AllocationResult:
        """Calculate allocations based on comprehensive performance"""

        if not performance_metrics:
            # Equal allocation fallback
            equal_share = Decimal("1.0") / Decimal(str(len(strategy_ids)))
            return AllocationResult(
                allocations={sid: equal_share for sid in strategy_ids},
                total_allocated=Decimal("1.0"),
                rebalance_required=False
            )

        # Calculate composite scores
        scores = {}
        for strategy_id in strategy_ids:
            metrics = performance_metrics.get(strategy_id, {})

            # Sharpe ratio component (normalize to 0-1)
            sharpe = metrics.get("sharpe_ratio", Decimal("0"))
            sharpe_normalized = self._normalize_sharpe(sharpe)
            sharpe_score = sharpe_normalized * self.sharpe_weight

            # Win rate component (already 0-1)
            win_rate = metrics.get("win_rate", Decimal("0")) / Decimal("100")
            win_rate_score = win_rate * self.win_rate_weight

            # Recent P&L component (normalize)
            recent_pnl = metrics.get("recent_pnl", Decimal("0"))
            pnl_normalized = self._normalize_pnl(recent_pnl, performance_metrics)
            pnl_score = pnl_normalized * self.recent_pnl_weight

            # Composite score
            scores[strategy_id] = sharpe_score + win_rate_score + pnl_score

        # Convert scores to allocations
        total_score = sum(scores.values())
        if total_score == 0:
            equal_share = Decimal("1.0") / Decimal(str(len(strategy_ids)))
            allocations = {sid: equal_share for sid in strategy_ids}
        else:
            allocations = {sid: score / total_score for sid, score in scores.items()}

        # Apply constraints
        allocations = self._apply_constraints(allocations)

        # Normalize
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}

        return AllocationResult(
            allocations=allocations,
            total_allocated=sum(allocations.values()),
            rebalance_required=True
        )

    def _normalize_sharpe(self, sharpe: Decimal) -> Decimal:
        """Normalize Sharpe ratio to 0-1 range"""

        # Assume Sharpe of 3.0 is excellent (maps to 1.0)
        # Negative Sharpe maps to 0
        if sharpe <= 0:
            return Decimal("0")

        normalized = sharpe / Decimal("3.0")
        return min(normalized, Decimal("1.0"))

    def _normalize_pnl(
        self,
        pnl: Decimal,
        all_metrics: Dict[str, Dict[str, Decimal]]
    ) -> Decimal:
        """Normalize P&L to 0-1 range based on all strategies"""

        all_pnls = [m.get("recent_pnl", Decimal("0")) for m in all_metrics.values()]

        if not all_pnls:
            return Decimal("0.5")

        max_pnl = max(all_pnls)
        min_pnl = min(all_pnls)

        if max_pnl == min_pnl:
            return Decimal("0.5")

        # Normalize to 0-1
        normalized = (pnl - min_pnl) / (max_pnl - min_pnl)
        return normalized

    def _apply_constraints(self, allocations: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """Apply min/max constraints"""

        constrained = {}

        for strategy_id, allocation in allocations.items():
            constrained[strategy_id] = max(
                self.min_allocation,
                min(self.max_allocation, allocation)
            )

        return constrained


logger.info("Capital allocator module loaded")
