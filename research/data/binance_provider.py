"""
Binance Data Provider

Implementation of IDataProvider for Binance exchange.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from loguru import logger
import ccxt.async_support as ccxt
import asyncio

from .interface import IDataProvider, MarketData


class BinanceDataProvider(IDataProvider):
    """
    Binance data provider using CCXT

    Single Responsibility: Fetch data from Binance
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize Binance data provider

        Args:
            testnet: Use testnet instead of mainnet
        """
        self.testnet = testnet
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # or 'future'
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"Initialized BinanceDataProvider (testnet={testnet})")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        exchange: str = "binance"
    ) -> MarketData:
        """
        Fetch OHLCV candlestick data from Binance

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe ("1m", "5m", "1h", "1d", etc.)
            start_date: Start date
            end_date: End date
            exchange: Exchange name (always binance for this provider)

        Returns:
            MarketData object with OHLCV DataFrame
        """
        logger.info(
            f"Fetching OHLCV: {symbol} {timeframe} "
            f"from {start_date.date()} to {end_date.date()}"
        )

        # Convert dates to timestamps
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)

        # Fetch data in chunks (Binance limits to 1000 candles per request)
        all_candles = []
        current_since = since

        while current_since < until:
            try:
                candles = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Update since to last candle timestamp + 1
                current_since = candles[-1][0] + 1

                # Stop if we've reached the end date
                if candles[-1][0] >= until:
                    break

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching OHLCV: {e}")
                break

        # Convert to DataFrame
        if not all_candles:
            logger.warning(f"No data fetched for {symbol} {timeframe}")
            return MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=pd.DataFrame(),
                start_date=start_date,
                end_date=end_date,
                source="binance"
            )

        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert to Decimal for price precision
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Filter to date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        logger.info(
            f"Fetched {len(df)} candles for {symbol} {timeframe}"
        )

        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=df,
            start_date=start_date,
            end_date=end_date,
            source="binance"
        )

    async def fetch_tick_data(
        self,
        symbol: str,
        date: datetime,
        exchange: str = "binance"
    ) -> pd.DataFrame:
        """
        Fetch tick-by-tick data for a single day

        Note: Binance doesn't provide tick data via CCXT easily.
        This would require using Binance Historical Data service.

        For now, we'll use 1m candles as approximation.
        """
        logger.warning(
            "Tick data not directly available via CCXT. "
            "Using 1m candles as approximation."
        )

        # Fetch 1m data for the day
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        market_data = await self.fetch_ohlcv(
            symbol=symbol,
            timeframe="1m",
            start_date=start,
            end_date=end,
            exchange=exchange
        )

        return market_data.data

    async def get_available_symbols(
        self,
        exchange: str = "binance"
    ) -> List[str]:
        """Get list of available trading symbols"""
        try:
            markets = await self.exchange.load_markets()
            symbols = [
                symbol for symbol, market in markets.items()
                if market['active'] and market['spot']  # Only active spot markets
            ]

            logger.info(f"Found {len(symbols)} available symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []

    async def get_latest_price(
        self,
        symbol: str,
        exchange: str = "binance"
    ) -> Decimal:
        """Get latest price for a symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            price = Decimal(str(ticker['last']))

            logger.debug(f"Latest price for {symbol}: {price}")
            return price

        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return Decimal("0")

    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()
        logger.info("Closed Binance connection")


logger.info("Binance data provider loaded")
