import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import pandas as pd

# Ensure local imports resolve when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from importlib.machinery import SourceFileLoader  # noqa: E402

import ccxt  # noqa: E402
import websockets  # noqa: E402

from indicators import AddIndicators  # noqa: E402
from utils import LogDiffMinMaxScaler, seed_everything  # noqa: E402


_rb7_module = None

def load_rb7():
    global _rb7_module
    if _rb7_module is None:
        _rb7_module = SourceFileLoader(
            "rb7_module", os.path.join(CURRENT_DIR, "RL-Bitcoin-trading-bot_7.py")
        ).load_module()
    return _rb7_module


def ohlcv_to_dataframe(ohlcvs):
    df = pd.DataFrame(ohlcvs, columns=["Date", "Open", "High", "Low", "Close", "Volume"])  # type: ignore
    # Convert timestamps to ISO strings for plotting
    df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


async def fetch_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str, since_ms: Optional[int], limit: int) -> pd.DataFrame:
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange '{exchange_id}' in ccxt")
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    all_rows = []
    fetch_since = since_ms
    while True:
        batch = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, timeframe, fetch_since, limit)
        if not batch:
            break
        all_rows.extend(batch)
        # pagination: next since is last open time + 1 ms
        last_ts = batch[-1][0]
        next_since = last_ts + 1
        # stop if no progress
        if fetch_since is not None and next_since <= fetch_since:
            break
        fetch_since = next_since
        # safety cap to avoid uncontrolled loops
        if len(all_rows) >= 50000:
            break
        # be nice to API
        await asyncio.sleep(exchange.rateLimit / 1000.0)

    if not all_rows:
        raise RuntimeError("No OHLCV data returned from exchange")

    df = ohlcv_to_dataframe(all_rows)
    # Basic sanity clean-up
    df = df.dropna().sort_values('Date')
    return df


async def fetch_ohlcv_with_fallback(exchanges: List[str], symbol: str, timeframe: str, since_ms: Optional[int], limit: int):
    last_error = None
    for ex in exchanges:
        try:
            return ex, await fetch_ohlcv_ccxt(ex, symbol, timeframe, since_ms, limit)
        except Exception as e:
            last_error = str(e)
            # try next exchange
            continue
    raise RuntimeError(f"All exchanges failed for {symbol} {timeframe}. Last error: {last_error}")


def prepare_datasets(df: pd.DataFrame, lookback: int, test_window: int):
    df_with_ind = AddIndicators(df.copy())
    # drop first 100 rows to avoid NaNs from indicators
    df_with_ind = df_with_ind[100:].dropna()

    depth = len(list(df_with_ind.columns[1:]))

    train_df = df_with_ind[:-test_window - lookback]
    test_df = df_with_ind[-test_window - lookback:]

    scaler = LogDiffMinMaxScaler().fit(train_df)
    train_df_norm = scaler.transform(train_df)
    test_df_norm = scaler.transform(test_df)

    return depth, train_df, test_df, train_df_norm, test_df_norm


async def handle_fetch(request: dict, websocket):
    exchange_id = request.get('exchange') or 'bybit'
    exchanges = request.get('exchanges') or [exchange_id, 'okx', 'kucoin', 'gate', 'bitget', 'bitfinex']
    symbol = request.get('symbol', 'WIF/USDT')
    timeframe = request.get('timeframe', '1h')
    hours = int(request.get('hours', 720))
    limit = int(request.get('limit', 1000))

    since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_ms = int(since_dt.timestamp() * 1000)

    await websocket.send(json.dumps({"status": "fetching", "details": {"exchanges": exchanges, "symbol": symbol, "timeframe": timeframe, "since": since_ms}}))
    try:
        used_exchange, df = await fetch_ohlcv_with_fallback(exchanges, symbol, timeframe, since_ms, limit)
    except Exception as e:
        await websocket.send(json.dumps({"status": "error", "error": str(e)}))
        return

    # Return summary only to avoid heavy payload
    summary = {
        "exchange": used_exchange,
        "rows": int(len(df)),
        "start": df.iloc[0]["Date"],
        "end": df.iloc[-1]["Date"],
        "columns": df.columns.tolist(),
    }
    await websocket.send(json.dumps({"status": "fetched", "summary": summary}))


async def handle_train(request: dict, websocket):
    # Defaults
    exchange_id = request.get('exchange') or 'bybit'
    exchanges = request.get('exchanges') or [exchange_id, 'okx', 'kucoin', 'gate', 'bitget', 'bitfinex']
    symbol = request.get('symbol', 'BTC/USDT')
    timeframe = request.get('timeframe', '1h')
    hours = int(request.get('hours', 720))  # history depth
    limit = int(request.get('limit', 1000))
    episodes = int(request.get('episodes', 100))
    lookback = int(request.get('lookback', 100))
    lr = float(request.get('lr', 1e-5))
    epochs = int(request.get('epochs', 5))
    batch_size = int(request.get('batch_size', 32))
    fee = float(request.get('fee', 0.001))
    model_kind = request.get('model', 'CNN')
    seed = int(request.get('seed', 42))
    finetune_folder = request.get('folder')
    finetune_name = request.get('name')

    seed_everything(seed)

    since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_ms = int(since_dt.timestamp() * 1000)

    await websocket.send(json.dumps({"status": "fetching", "details": {"exchanges": exchanges, "symbol": symbol, "timeframe": timeframe, "since": since_ms}}))
    try:
        used_exchange, df = await fetch_ohlcv_with_fallback(exchanges, symbol, timeframe, since_ms, limit)
    except Exception as e:
        await websocket.send(json.dumps({"status": "error", "error": str(e)}))
        return

    await websocket.send(json.dumps({"status": "preparing"}))
    depth, train_df, test_df, train_df_norm, test_df_norm = prepare_datasets(df, lookback=lookback, test_window=min(len(df)//4, 720))

    await websocket.send(json.dumps({"status": "training", "episodes": episodes, "exchange": used_exchange}))

    # Lazy import RL module and optimizer only when model is about to be built
    rb7 = load_rb7()
    CustomAgent = rb7.CustomAgent
    CustomEnv = rb7.CustomEnv
    train_agent = rb7.train_agent
    Adam = rb7.Adam

    agent = CustomAgent(lookback_window_size=lookback, lr=lr, epochs=epochs, optimizer=Adam, batch_size=batch_size, model=model_kind, depth=depth, comment=f"ccxt:{used_exchange}:{symbol}:{timeframe}")

    if finetune_folder and finetune_name:
        try:
            agent.load(finetune_folder, finetune_name)
            await websocket.send(json.dumps({"status": "finetune_loaded", "folder": finetune_folder, "name": finetune_name}))
        except Exception as e:
            await websocket.send(json.dumps({"status": "finetune_load_failed", "error": str(e)}))

    env = CustomEnv(df=train_df, df_normalized=train_df_norm, lookback_window_size=lookback, step_reward='net_worth_delta', trade_fee=fee)

    # Run training in a thread to avoid blocking the event loop
    def _train_sync():
        train_agent(env, agent, visualize=False, train_episodes=episodes, training_batch_size=500)

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _train_sync)
        await websocket.send(json.dumps({"status": "done", "result": "trained", "log_dir": getattr(agent, 'log_name', None)}))
    except Exception as e:
        await websocket.send(json.dumps({"status": "error", "error": str(e)}))


async def serve(websocket):
    try:
        async for message in websocket:
            try:
                req = json.loads(message)
            except Exception:
                await websocket.send(json.dumps({"status": "error", "error": "Invalid JSON"}))
                continue

            action = req.get('action')
            if action == 'train':
                await handle_train(req, websocket)
            elif action == 'fetch':
                await handle_fetch(req, websocket)
            else:
                await websocket.send(json.dumps({"status": "error", "error": "Unknown action"}))
    except websockets.ConnectionClosed:
        return


async def run_server():
    host = os.environ.get('WS_HOST', '0.0.0.0')
    port = int(os.environ.get('WS_PORT', '8765'))
    async with websockets.serve(serve, host, port, max_size=8 * 1024 * 1024):
        print(f"WebSocket server listening on ws://{host}:{port}")
        # Run forever
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(run_server())