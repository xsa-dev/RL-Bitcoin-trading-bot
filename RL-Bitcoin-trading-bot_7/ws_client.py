import asyncio
import json
import os

import websockets


async def main():
    host = os.environ.get('WS_HOST', '127.0.0.1')
    port = int(os.environ.get('WS_PORT', '8765'))
    uri = f"ws://{host}:{port}"
    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        # Example: fetch only
        req = {
            "action": "fetch",
            "exchange": "bybit",
            "symbol": "WIF/USDT",
            "timeframe": "1h",
            "hours": 720,
            "limit": 1000,
        }
        await ws.send(json.dumps(req))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                print(msg)
                data = json.loads(msg)
                if data.get("status") in {"fetched", "error"}:
                    break
        except asyncio.TimeoutError:
            print("Timed out waiting for server response.")


if __name__ == "__main__":
    asyncio.run(main())