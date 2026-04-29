import asyncio
import requests
from core_logic.session_logger import slog

SOUL_URL = "http://127.0.0.1:8001/soul"
VRAM_THRESHOLD_GB = 3.5
POLL_INTERVAL_SEC = 10


class VRAMSentinel:

    async def _check_vram(self) -> float:
        try:
            resp = requests.get(SOUL_URL, timeout=2)
            resp.raise_for_status()
            data = resp.json()
            gpu_str = data["vitals"]["gpu"]
            # Parse "3.7GB / 8.0GB" -> 3.7
            vram_used = float(gpu_str.split("GB /")[0].strip().replace("GB", ""))
            return vram_used
        except Exception as e:
            slog.warning(f"VRAM check failed: {e}")
            return 0.0

    async def monitor(self):
        while True:
            vram = await self._check_vram()
            if vram > VRAM_THRESHOLD_GB:
                print(f"🚨 VRAM ALERT: {vram:.2f}GB exceeds {VRAM_THRESHOLD_GB}GB threshold!")
                slog.warning(f"VRAM exceeded threshold: {vram:.2f}GB")
            await asyncio.sleep(POLL_INTERVAL_SEC)

    def start(self, event_loop=None):
        loop = event_loop or asyncio.get_event_loop()
        loop.create_task(self.monitor())
        slog.info("[VRAMSENTINEL] Started. Polling every 10s.")


# Integration: In api.py lifespan:
# vram_sentinel = VRAMSentinel()
# vram_sentinel.start()
