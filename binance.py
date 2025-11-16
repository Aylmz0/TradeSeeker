"""
Binance Futures order execution utilities for Alpha Arena.

Provides a thin REST client with request signing plus a higher-level executor
that converts bot decisions into exchange orders and synchronized portfolio
snapshots.
"""
from __future__ import annotations

import hmac
import hashlib
import json
import logging
import math
import time
import urllib.parse
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import Config

# Increase decimal precision for quantity rounding
getcontext().prec = 18

logger = logging.getLogger("AlphaArena.Binance")


class BinanceAPIError(Exception):
    """Exception raised when Binance returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@dataclass
class SymbolFilters:
    """Trading filters for a single futures symbol."""

    symbol: str
    step_size: float
    min_qty: float
    tick_size: float
    min_notional: float
    leverage_brackets: Optional[List[Dict[str, Any]]] = None


class BinanceFuturesClient:
    """Low-level REST client for Binance USDT-M futures endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet: bool = False,
        recv_window: int = 5000,
        timeout: int = 30,
    ):
        self.api_key = api_key or Config.BINANCE_API_KEY
        self.secret_key = secret_key or Config.BINANCE_SECRET_KEY
        if not self.api_key or not self.secret_key:
            raise BinanceAPIError("Binance API key/secret not configured for live trading.")

        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        self.recv_window = recv_window
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        self.session = requests.Session()

    # --- Internal helpers -------------------------------------------------
    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, payload: Dict[str, Any]) -> str:
        query_string = urllib.parse.urlencode(payload, doseq=True)
        signature = hmac.new(self.secret_key.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        params = params.copy() if params else {}
        headers = {"X-MBX-APIKEY": self.api_key}

        if signed:
            params["timestamp"] = self._timestamp()
            params["recvWindow"] = self.recv_window
            signature_payload = params.copy()
            signature = self._sign(signature_payload)
            params["signature"] = signature

        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, params=params, timeout=self.timeout, headers=headers)
        except requests.RequestException as exc:
            raise BinanceAPIError(f"HTTP request failed: {exc}") from exc

        if response.status_code >= 400:
            try:
                data = response.json()
                message = data.get("msg", response.text)
                error_code = data.get("code")
            except ValueError:
                message = response.text
                error_code = None
            raise BinanceAPIError(message, status_code=response.status_code, error_code=error_code)

        try:
            return response.json()
        except ValueError as exc:
            raise BinanceAPIError(f"Failed to decode JSON response: {exc}") from exc

    # --- Public REST wrappers ---------------------------------------------
    def get_exchange_info(self, symbols: List[str]) -> Dict[str, Any]:
        payload = {"symbols": json.dumps(symbols, separators=(",", ":"))}
        return self._request("GET", "/fapi/v1/exchangeInfo", params=payload, signed=False)

    def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})

    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        payload = {"symbol": symbol, "leverage": leverage}
        return self._request("POST", "/fapi/v1/leverage", params=payload, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        payload = {"symbol": symbol, "marginType": margin_type}
        return self._request("POST", "/fapi/v1/marginType", params=payload, signed=True)

    def place_order(self, payload: Dict[str, Any], test: bool = False) -> Dict[str, Any]:
        endpoint = "/fapi/v1/order/test" if test else "/fapi/v1/order"
        return self._request("POST", endpoint, params=payload, signed=True)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)

    def get_balance(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def get_account_info(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_position_risk(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v2/positionRisk", signed=True)


class BinanceOrderExecutor:
    """High-level executor that handles sizing, rounding, and synchronization."""

    def __init__(self, coins: List[str]):
        self.coins = coins
        self.symbol_map = {coin: f"{coin}USDT" for coin in coins}
        self.live = Config.TRADING_MODE == "live"
        self.testnet = Config.BINANCE_TESTNET
        self.margin_type = Config.BINANCE_MARGIN_TYPE.upper()
        self.default_leverage = Config.BINANCE_DEFAULT_LEVERAGE
        self.recv_window = Config.BINANCE_RECV_WINDOW

        self.client: Optional[BinanceFuturesClient] = None
        self.symbol_filters: Dict[str, SymbolFilters] = {}
        self.symbol_leverage: Dict[str, int] = {}

        if not self.live:
            logger.info("BinanceOrderExecutor initialized in simulation mode.")
            return

        self._initialize_client()
        self._load_symbol_filters()
        self._ensure_symbol_settings()

    # --- Initialization helpers ------------------------------------------
    def _initialize_client(self):
        try:
            self.client = BinanceFuturesClient(
                api_key=Config.BINANCE_API_KEY,
                secret_key=Config.BINANCE_SECRET_KEY,
                testnet=self.testnet,
                recv_window=self.recv_window,
                timeout=Config.REQUEST_TIMEOUT,
            )
            logger.info("Binance futures client initialized. Testnet=%s", self.testnet)
        except BinanceAPIError as exc:
            logger.error("Failed to initialize Binance client: %s", exc)
            raise

    def _load_symbol_filters(self):
        if not self.client:
            return
        symbols = list(self.symbol_map.values())
        info = self.client.get_exchange_info(symbols)
        filters = {}
        for symbol_info in info.get("symbols", []):
            symbol = symbol_info["symbol"]
            filt_map = {f["filterType"]: f for f in symbol_info.get("filters", [])}
            lot_filter = filt_map.get("LOT_SIZE", {})
            price_filter = filt_map.get("PRICE_FILTER", {})
            min_notional_filter = filt_map.get("MIN_NOTIONAL", {})

            filters[symbol] = SymbolFilters(
                symbol=symbol,
                step_size=float(lot_filter.get("stepSize", "0.0001")),
                min_qty=float(lot_filter.get("minQty", "0.0")),
                tick_size=float(price_filter.get("tickSize", "0.0001")),
                min_notional=float(min_notional_filter.get("notional", "0.0")),
                leverage_brackets=symbol_info.get("brackets"),
            )
        self.symbol_filters = filters
        logger.info("Loaded Binance filters for symbols: %s", ", ".join(filters.keys()))

    def _ensure_symbol_settings(self):
        if not self.client:
            return
        for coin, symbol in self.symbol_map.items():
            try:
                if self.margin_type in {"ISOLATED", "CROSSED"}:
                    try:
                        self.client.set_margin_type(symbol, self.margin_type)
                        logger.info("Set %s margin type to %s", symbol, self.margin_type)
                    except BinanceAPIError as exc:
                        # Error code -4046 occurs if margin type already set
                        if exc.error_code not in (-4046, -4097):
                            logger.warning("Margin type update failed for %s: %s", symbol, exc)
                self.client.change_leverage(symbol, self.default_leverage)
                self.symbol_leverage[symbol] = self.default_leverage
            except BinanceAPIError as exc:
                logger.warning("Leverage setup failed for %s (%s): %s", coin, symbol, exc)

    # --- Utility methods --------------------------------------------------
    def _round_to_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        quantized = (d_value / d_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * d_step
        return float(quantized)

    def _validate_and_format_quantity(self, symbol: str, quantity: float, price: float) -> float:
        if quantity <= 0:
            raise BinanceAPIError(f"Quantity {quantity} invalid for {symbol}")

        filters = self.symbol_filters.get(symbol)
        if not filters:
            return quantity

        adjusted_qty = self._round_to_step(quantity, filters.step_size)
        if adjusted_qty < filters.min_qty:
            raise BinanceAPIError(
                f"Quantity {adjusted_qty} below minimum {filters.min_qty} for {symbol}",
            )

        if price and filters.min_notional > 0:
            notional = adjusted_qty * price
            if notional < filters.min_notional:
                raise BinanceAPIError(
                    f"Notional ${notional:.4f} below minimum ${filters.min_notional:.2f} for {symbol}",
                )
        return adjusted_qty

    def _ensure_leverage(self, symbol: str, leverage: int):
        leverage = int(max(1, leverage))
        current = self.symbol_leverage.get(symbol)
        if current == leverage:
            return
        assert self.client is not None
        self.client.change_leverage(symbol, leverage)
        self.symbol_leverage[symbol] = leverage

    def _determine_side(self, direction: str, action: str) -> str:
        """
        Determine Binance side parameter.

        direction: 'long' / 'short'
        action: 'open' / 'close'
        """
        if action == "open":
            return "BUY" if direction == "long" else "SELL"
        else:
            return "SELL" if direction == "long" else "BUY"

    def _extract_fill_details(self, order: Dict[str, Any]) -> Tuple[float, float]:
        executed_qty = float(order.get("executedQty", order.get("origQty", "0")))
        avg_price = float(order.get("avgPrice", 0.0))
        if executed_qty > 0 and (avg_price == 0.0 or math.isclose(avg_price, 0.0)):
            cum_quote = float(order.get("cumQuote", 0.0))
            if cum_quote > 0:
                avg_price = cum_quote / executed_qty
        return executed_qty, avg_price

    # --- Public API -------------------------------------------------------
    def is_live(self) -> bool:
        return self.live and self.client is not None

    def get_account_overview(self) -> Dict[str, float]:
        """Get account overview including total wallet balance (equity) from Binance."""
        if not self.is_live():
            return {}
        assert self.client is not None
        
        overview = {
            "availableBalance": 0.0,
            "walletBalance": 0.0,
            "totalWalletBalance": 0.0
        }
        
        # Use /fapi/v2/account to get total wallet balance (includes unrealized PnL)
        # This endpoint returns totalWalletBalance which = walletBalance + totalUnrealizedProfit
        account_info = self.client.get_account_info()
        
        if account_info:
            # Binance /fapi/v2/account returns these fields:
            # - totalWalletBalance: Total equity (wallet balance + unrealized PnL)
            # - availableBalance: Available balance for trading
            # - walletBalance: Wallet balance (without unrealized PnL)
            # - totalUnrealizedProfit: Total unrealized profit/loss
            
            # Try different possible field names (Binance uses totalWalletBalance)
            total_wallet_balance = (
                account_info.get("totalWalletBalance") or
                account_info.get("totalEquity")
            )
            
            if total_wallet_balance is not None:
                try:
                    overview["totalWalletBalance"] = float(total_wallet_balance)
                except (ValueError, TypeError):
                    pass
            
            # Also get availableBalance and walletBalance from account info
            available = account_info.get("availableBalance")
            wallet_bal = account_info.get("walletBalance")
            
            if available is not None:
                try:
                    overview["availableBalance"] = float(available)
                except (ValueError, TypeError):
                    pass
            
            if wallet_bal is not None:
                try:
                    overview["walletBalance"] = float(wallet_bal)
                except (ValueError, TypeError):
                    pass
        
        # Fallback: Also get available balance from balance endpoint
        # This is needed if account_info doesn't have availableBalance
        if overview["availableBalance"] == 0.0:
            try:
                balances = self.client.get_balance()
                for asset in balances:
                    if asset.get("asset") == "USDT":
                        available_bal = asset.get("availableBalance", 0.0)
                        if available_bal:
                            overview["availableBalance"] = float(available_bal)
                        # walletBalance is just the USDT balance without unrealized PnL
                        balance = asset.get("balance", 0.0)
                        if balance and overview["walletBalance"] == 0.0:
                            overview["walletBalance"] = float(balance)
                        break
            except Exception:
                pass
        
        return overview

    def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
        if not self.is_live():
            return {}
        assert self.client is not None
        raw_positions = self.client.get_position_risk()
        snapshot: Dict[str, Dict[str, Any]] = {}
        for entry in raw_positions:
            symbol = entry.get("symbol")
            coin = next((c for c, s in self.symbol_map.items() if s == symbol), None)
            if not coin:
                continue
            position_amt = float(entry.get("positionAmt", 0.0))
            if math.isclose(position_amt, 0.0, abs_tol=1e-10):
                continue
            entry_price = float(entry.get("entryPrice", 0.0))
            mark_price = float(entry.get("markPrice", entry_price))
            direction = "long" if position_amt > 0 else "short"
            quantity = abs(position_amt)
            leverage = int(float(entry.get("leverage", self.default_leverage)))
            unrealized_pnl = float(entry.get("unRealizedProfit", 0.0))
            isolated_margin = entry.get("isolatedMargin")
            margin_usd = float(isolated_margin) if isolated_margin is not None else abs(quantity * mark_price / max(leverage, 1))

            snapshot[coin] = {
                "symbol": coin,
                "direction": direction,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": mark_price,
                "unrealized_pnl": unrealized_pnl,
                "notional_usd": quantity * mark_price,
                "margin_usd": margin_usd,
                "leverage": leverage,
                "entry_time": entry.get("updateTime"),
                "liquidation_price": float(entry.get("liquidationPrice", 0.0)),
                "confidence": 0.0,
                "exit_plan": {},
                "loss_cycle_count": 0,
                "risk_usd": margin_usd,
                "exchange_metadata": {
                    "symbol": symbol,
                    "raw": entry,
                },
            }
        return snapshot

    def place_market_order(
        self,
        coin: str,
        direction: str,
        quantity: float,
        leverage: int,
        price_reference: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        if not self.is_live():
            raise BinanceAPIError("Attempted to place live order while executor disabled.")

        symbol = self.symbol_map[coin]
        self._ensure_leverage(symbol, leverage)
        adjusted_qty = self._validate_and_format_quantity(symbol, quantity, price_reference)

        payload = {
            "symbol": symbol,
            "side": self._determine_side(direction, "close" if reduce_only else "open"),
            "type": "MARKET",
            "quantity": adjusted_qty,
            "reduceOnly": "true" if reduce_only else "false",
            "newOrderRespType": "RESULT",
        }

        logger.info(
            "Placing market order %s %s qty=%s reduceOnly=%s",
            symbol,
            payload["side"],
            adjusted_qty,
            reduce_only,
        )

        assert self.client is not None
        order = self.client.place_order(payload)
        executed_qty, avg_price = self._extract_fill_details(order)
        if executed_qty == 0:
            raise BinanceAPIError(f"Order returned zero fills: {order}")

        order["executedQty"] = executed_qty
        order["avgPriceComputed"] = avg_price
        return order

    def close_position(
        self,
        coin: str,
        direction: str,
        quantity: float,
        price_reference: float,
    ) -> Dict[str, Any]:
        return self.place_market_order(
            coin=coin,
            direction=direction,
            quantity=quantity,
            leverage=self.symbol_leverage.get(self.symbol_map[coin], self.default_leverage),
            price_reference=price_reference,
            reduce_only=True,
        )

    def place_take_profit_order(
        self,
        coin: str,
        direction: str,
        stop_price: float,
        quantity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a TAKE_PROFIT_MARKET order on Binance.
        
        Args:
            coin: Coin symbol (e.g., 'BTC')
            direction: 'long' or 'short'
            stop_price: Price at which to trigger the take profit
            quantity: Optional quantity (if None, uses closePosition=true)
        
        Returns:
            Order response from Binance
        """
        if not self.is_live():
            raise BinanceAPIError("Attempted to place live order while executor disabled.")

        symbol = self.symbol_map[coin]
        filters = self.symbol_filters.get(symbol)
        if filters:
            stop_price = self._round_to_step(stop_price, filters.tick_size)

        # For take profit: long positions trigger when price goes UP, short when price goes DOWN
        # So for long: stopPrice should be above current price, side should be SELL
        # For short: stopPrice should be below current price, side should be BUY
        side = "SELL" if direction == "long" else "BUY"

        payload = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": stop_price,
            "newOrderRespType": "RESULT",
        }

        if quantity is not None:
            adjusted_qty = self._validate_and_format_quantity(symbol, quantity, stop_price)
            payload["quantity"] = adjusted_qty
        else:
            payload["closePosition"] = "true"

        logger.info(
            "Placing take profit order %s %s stopPrice=%s quantity=%s",
            symbol,
            side,
            stop_price,
            quantity if quantity else "closePosition",
        )

        assert self.client is not None
        return self.client.place_order(payload)

    def place_stop_loss_order(
        self,
        coin: str,
        direction: str,
        stop_price: float,
        quantity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a STOP_MARKET order on Binance.
        
        Args:
            coin: Coin symbol (e.g., 'BTC')
            direction: 'long' or 'short'
            stop_price: Price at which to trigger the stop loss
            quantity: Optional quantity (if None, uses closePosition=true)
        
        Returns:
            Order response from Binance
        """
        if not self.is_live():
            raise BinanceAPIError("Attempted to place live order while executor disabled.")

        symbol = self.symbol_map[coin]
        filters = self.symbol_filters.get(symbol)
        if filters:
            stop_price = self._round_to_step(stop_price, filters.tick_size)

        # For stop loss: long positions trigger when price goes DOWN, short when price goes UP
        # So for long: stopPrice should be below current price, side should be SELL
        # For short: stopPrice should be above current price, side should be BUY
        side = "SELL" if direction == "long" else "BUY"

        payload = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "newOrderRespType": "RESULT",
        }

        if quantity is not None:
            adjusted_qty = self._validate_and_format_quantity(symbol, quantity, stop_price)
            payload["quantity"] = adjusted_qty
        else:
            payload["closePosition"] = "true"

        logger.info(
            "Placing stop loss order %s %s stopPrice=%s quantity=%s",
            symbol,
            side,
            stop_price,
            quantity if quantity else "closePosition",
        )

        assert self.client is not None
        return self.client.place_order(payload)

    def cancel_all_orders_for_symbol(self, coin: str) -> Dict[str, Any]:
        """Cancel all open orders (including TP/SL) for a symbol."""
        if not self.is_live():
            raise BinanceAPIError("Attempted to cancel orders while executor disabled.")
        
        symbol = self.symbol_map[coin]
        assert self.client is not None
        return self.client.cancel_all_orders(symbol)

