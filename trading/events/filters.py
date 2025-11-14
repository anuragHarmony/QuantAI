"""
Event filters for fine-grained event subscription

Filters allow subscribers to receive only events that match specific criteria.
Follows composition pattern for combining multiple filters.

Example:
    # Subscribe to BTC ticks from Binance only
    filter = and_filter(
        symbol_filter("BTC/USDT"),
        exchange_filter("binance")
    )
    await bus.subscribe("tick", handler, filter_func=filter)
"""
from typing import Callable
from datetime import datetime

from .base import BaseEvent


# Type alias for filter functions
EventFilter = Callable[[BaseEvent], bool]


def symbol_filter(symbol: str) -> EventFilter:
    """
    Filter events by symbol

    Args:
        symbol: Normalized symbol (e.g., "BTC/USDT")

    Returns:
        Filter function
    """
    def filter_func(event: BaseEvent) -> bool:
        return event.symbol == symbol

    return filter_func


def exchange_filter(exchange: str) -> EventFilter:
    """
    Filter events by exchange

    Args:
        exchange: Exchange name (e.g., "binance")

    Returns:
        Filter function
    """
    def filter_func(event: BaseEvent) -> bool:
        return event.exchange == exchange

    return filter_func


def event_type_filter(event_type: str) -> EventFilter:
    """
    Filter events by type

    Args:
        event_type: Event type (e.g., "tick", "order_filled")

    Returns:
        Filter function
    """
    def filter_func(event: BaseEvent) -> bool:
        return event.event_type == event_type

    return filter_func


def time_range_filter(start: datetime, end: datetime) -> EventFilter:
    """
    Filter events by time range

    Args:
        start: Start timestamp
        end: End timestamp

    Returns:
        Filter function
    """
    def filter_func(event: BaseEvent) -> bool:
        return start <= event.timestamp <= end

    return filter_func


def strategy_filter(strategy_id: str) -> EventFilter:
    """
    Filter events by strategy ID

    Args:
        strategy_id: Strategy identifier

    Returns:
        Filter function
    """
    def filter_func(event: BaseEvent) -> bool:
        # Check if event has strategy_id attribute
        return getattr(event, "strategy_id", None) == strategy_id

    return filter_func


def and_filter(*filters: EventFilter) -> EventFilter:
    """
    Combine filters with AND logic

    All filters must pass for event to match.

    Args:
        *filters: Filter functions to combine

    Returns:
        Combined filter function

    Example:
        filter = and_filter(
            symbol_filter("BTC/USDT"),
            exchange_filter("binance")
        )
    """
    def filter_func(event: BaseEvent) -> bool:
        return all(f(event) for f in filters)

    return filter_func


def or_filter(*filters: EventFilter) -> EventFilter:
    """
    Combine filters with OR logic

    Any filter can pass for event to match.

    Args:
        *filters: Filter functions to combine

    Returns:
        Combined filter function

    Example:
        filter = or_filter(
            symbol_filter("BTC/USDT"),
            symbol_filter("ETH/USDT")
        )
    """
    def filter_func(event: BaseEvent) -> bool:
        return any(f(event) for f in filters)

    return filter_func


def not_filter(filter_to_negate: EventFilter) -> EventFilter:
    """
    Negate a filter

    Args:
        filter_to_negate: Filter to negate

    Returns:
        Negated filter function

    Example:
        # All events except BTC
        filter = not_filter(symbol_filter("BTC/USDT"))
    """
    def filter_func(event: BaseEvent) -> bool:
        return not filter_to_negate(event)

    return filter_func


def symbols_filter(symbols: list[str]) -> EventFilter:
    """
    Filter events by multiple symbols

    Args:
        symbols: List of symbols

    Returns:
        Filter function

    Example:
        filter = symbols_filter(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    """
    symbol_set = set(symbols)

    def filter_func(event: BaseEvent) -> bool:
        return event.symbol in symbol_set

    return filter_func


def exchanges_filter(exchanges: list[str]) -> EventFilter:
    """
    Filter events by multiple exchanges

    Args:
        exchanges: List of exchanges

    Returns:
        Filter function
    """
    exchange_set = set(exchanges)

    def filter_func(event: BaseEvent) -> bool:
        return event.exchange in exchange_set

    return filter_func


def attribute_filter(attribute: str, value: any) -> EventFilter:
    """
    Generic filter by any attribute

    Args:
        attribute: Attribute name
        value: Expected value

    Returns:
        Filter function

    Example:
        # Filter by order side
        filter = attribute_filter("side", "buy")
    """
    def filter_func(event: BaseEvent) -> bool:
        return getattr(event, attribute, None) == value

    return filter_func


def custom_filter(predicate: Callable[[BaseEvent], bool]) -> EventFilter:
    """
    Create custom filter from predicate function

    Args:
        predicate: Custom predicate function

    Returns:
        Filter function

    Example:
        # Filter ticks with spread > 0.1%
        def high_spread(event):
            if hasattr(event, 'spread') and hasattr(event, 'mid_price'):
                return (event.spread / event.mid_price) > 0.001
            return False

        filter = custom_filter(high_spread)
    """
    return predicate


# Convenience composite filters

def binance_btc_filter() -> EventFilter:
    """Filter for BTC/USDT on Binance"""
    return and_filter(
        exchange_filter("binance"),
        symbol_filter("BTC/USDT")
    )


def major_pairs_filter() -> EventFilter:
    """Filter for major trading pairs"""
    return symbols_filter([
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT"
    ])
