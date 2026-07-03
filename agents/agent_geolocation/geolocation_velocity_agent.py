from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import structlog
from prometheus_client import start_http_server

from agents.base_agent import BaseSwarmAgent
from shared.transaction_schema import TransactionPayload

logger = structlog.get_logger()

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat1_rad, lng1_rad, lat2_rad, lng2_rad = map(radians, (lat1, lng1, lat2, lng2))
    delta_lat = lat2_rad - lat1_rad
    delta_lng = lng2_rad - lng1_rad
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c


class GeolocationVelocityAgent(BaseSwarmAgent):
    def __init__(
        self,
        redis_url: str,
        input_channel: str,
        output_channel: str,
        threshold: float,
        max_plausible_velocity_kmh: float = 900.0,
        max_tracked_users: int = 1_000_000,
    ) -> None:
        super().__init__(
            agent_id="agent_geolocation",
            redis_url=redis_url,
            input_channel=input_channel,
            output_channel=output_channel,
            threshold=threshold,
            feature_used="velocity_kmh",
        )
        self._max_plausible_velocity_kmh = max_plausible_velocity_kmh
        self._max_tracked_users = max_tracked_users
        self._last_position: OrderedDict[str, tuple[float, float, datetime]] = OrderedDict()

    def _record_position(self, user_id: str, lat: float, lng: float, ts: datetime) -> None:
        self._last_position[user_id] = (lat, lng, ts)
        self._last_position.move_to_end(user_id)
        if len(self._last_position) > self._max_tracked_users:
            self._last_position.popitem(last=False)

    async def compute_score(self, tx: TransactionPayload) -> float:
        previous = self._last_position.get(tx.user_id)

        if previous is None:
            self._record_position(tx.user_id, tx.latitude, tx.longitude, tx.timestamp)
            return 0.0

        prev_lat, prev_lng, prev_timestamp = previous
        elapsed_seconds = (tx.timestamp - prev_timestamp).total_seconds()

        if elapsed_seconds < 0:
            logger.warning(
                "out_of_order_event",
                agent_id=self.agent_id,
                user_id=tx.user_id,
                transaction_id=tx.transaction_id,
            )
            return 0.0

        distance_km = haversine_distance_km(prev_lat, prev_lng, tx.latitude, tx.longitude)
        self._record_position(tx.user_id, tx.latitude, tx.longitude, tx.timestamp)

        if elapsed_seconds == 0:
            return 1.0 if distance_km > 0.0 else 0.0

        velocity_kmh = distance_km / (elapsed_seconds / 3600.0)
        return min(velocity_kmh / self._max_plausible_velocity_kmh, 1.0)


def build_agent_from_env() -> GeolocationVelocityAgent:
    return GeolocationVelocityAgent(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        input_channel=os.environ.get("TX_EVENTS_CHANNEL", "tx_events"),
        output_channel=os.environ.get("AGENT_SCORES_CHANNEL", "agent_scores"),
        threshold=float(os.environ.get("THRESHOLD_AGENT_GEOLOCATION", "0.80")),
    )


async def main() -> None:
    start_http_server(int(os.environ.get("METRICS_PORT", "8003")))
    agent = build_agent_from_env()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())