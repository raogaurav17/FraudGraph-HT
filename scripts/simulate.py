"""
Demo transaction simulator for FraudGraph.
Generates a realistic stream of transactions with varying fraud rates
to populate the dashboard without real dataset files.

Usage:
    python scripts/simulate.py --count 500 --fraud-rate 0.04
    python scripts/simulate.py --stream --interval 1.0
"""

import asyncio
import random
import uuid
import math
import argparse
import httpx
from datetime import datetime, timezone

API_URL = "http://localhost:8000/api"

CHANNELS = ["online", "pos", "atm", "mobile"]
CATEGORIES = ["electronics", "travel", "grocery", "fuel", "entertainment", "services"]
DATASETS = ["ieee_cis", "paysim", "elliptic", "live"]

# Fraud profiles — (probability of selection, amount range, hour range, velocity_1h, fraud_rate)
PROFILES = [
    # Legit daytime purchase
    {"weight": 0.60, "amount": (5, 200),    "hour": (8, 20),   "v1h": (0, 3),  "mfr": 0.002, "cross": False},
    # Legit large purchase
    {"weight": 0.15, "amount": (200, 2000), "hour": (9, 18),   "v1h": (0, 2),  "mfr": 0.003, "cross": False},
    # Low-risk late night
    {"weight": 0.10, "amount": (10, 100),   "hour": (21, 23),  "v1h": (0, 2),  "mfr": 0.005, "cross": False},
    # Suspicious online (high velocity, late)
    {"weight": 0.06, "amount": (500, 5000), "hour": (0, 4),    "v1h": (5, 15), "mfr": 0.08,  "cross": True},
    # Fraud ring (very high velocity, large amount, cross-border)
    {"weight": 0.04, "amount": (2000, 20000),"hour": (1, 5),   "v1h": (12, 30),"mfr": 0.35,  "cross": True},
    # Card testing (tiny amounts, high velocity)
    {"weight": 0.05, "amount": (0.5, 5),    "hour": (2, 6),    "v1h": (10, 25),"mfr": 0.25,  "cross": False},
]

CARD_IDS = [f"CARD_{uuid.uuid4().hex[:8].upper()}" for _ in range(200)]
MERCHANT_IDS = [f"MERCH_{uuid.uuid4().hex[:6].upper()}" for _ in range(80)]
DEVICE_IDS = [f"DEV_{uuid.uuid4().hex[:6].upper()}" for _ in range(120)]


def weighted_choice(profiles):
    weights = [p["weight"] for p in profiles]
    total = sum(weights)
    r = random.random() * total
    for p, w in zip(profiles, weights):
        r -= w
        if r <= 0:
            return p
    return profiles[-1]


def generate_transaction(dataset_source: str = "live") -> dict:
    profile = weighted_choice(PROFILES)
    amount = round(random.uniform(*profile["amount"]), 2)
    hour = random.randint(*profile["hour"])
    v1h = random.randint(*profile["v1h"])

    return {
        "transaction_id": f"SIM_{uuid.uuid4().hex[:16].upper()}",
        "amount": amount,
        "channel": random.choice(CHANNELS),
        "product_category": random.choice(CATEGORIES),
        "card_id": random.choice(CARD_IDS),
        "merchant_id": random.choice(MERCHANT_IDS),
        "device_id": random.choice(DEVICE_IDS),
        "hour_of_day": hour,
        "velocity_1h": v1h,
        "velocity_24h": v1h * random.randint(3, 8),
        "country_mismatch": profile["cross"],
        "merchant_fraud_rate": profile["mfr"] + random.uniform(-0.005, 0.01),
        "card_avg_amount_30d": amount * random.uniform(0.5, 2.0),
        "dataset_source": dataset_source,
    }


async def send_transaction(client: httpx.AsyncClient, txn: dict) -> dict | None:
    try:
        r = await client.post(f"{API_URL}/predict", json=txn, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Error: {e}")
        return None


async def simulate_batch(count: int = 100, concurrency: int = 10, dataset: str = "live"):
    print(f"Simulating {count} transactions (concurrency={concurrency})...")
    results = {"total": 0, "fraud": 0, "errors": 0, "risk": {}}

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(i: int):
            async with semaphore:
                ds = random.choice(DATASETS) if dataset == "mixed" else dataset
                txn = generate_transaction(ds)
                result = await send_transaction(client, txn)
                if result:
                    prob = result.get("fraud_probability", 0)
                    risk = result.get("risk_level", "LOW")
                    results["total"] += 1
                    results["risk"][risk] = results["risk"].get(risk, 0) + 1
                    if risk in ("HIGH", "CRITICAL"):
                        results["fraud"] += 1
                    if (i + 1) % 50 == 0:
                        print(f"  [{i+1}/{count}] fraud_rate={results['fraud']/max(results['total'],1):.2%}")
                else:
                    results["errors"] += 1

        tasks = [process_one(i) for i in range(count)]
        await asyncio.gather(*tasks)

    print(f"\nDone: {results['total']} scored, {results['fraud']} fraud flagged")
    print(f"Risk distribution: {results['risk']}")
    print(f"Errors: {results['errors']}")
    return results


async def simulate_stream(interval: float = 1.0, dataset: str = "mixed"):
    """Continuous streaming simulation for live dashboard demo."""
    print(f"Streaming transactions every {interval}s... (Ctrl+C to stop)")
    count = 0
    async with httpx.AsyncClient() as client:
        while True:
            try:
                ds = random.choice(DATASETS) if dataset == "mixed" else dataset
                txn = generate_transaction(ds)
                result = await send_transaction(client, txn)
                if result:
                    count += 1
                    risk = result.get("risk_level", "LOW")
                    prob = result.get("fraud_probability", 0)
                    icon = "🔴" if risk == "CRITICAL" else "🟠" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
                    print(f"{icon} [{count:05d}] {txn['amount']:>10.2f} USD | {prob:.3f} | {risk:<8} | {ds}")
                await asyncio.sleep(interval + random.uniform(-0.1, 0.2))
            except KeyboardInterrupt:
                print(f"\nStopped after {count} transactions")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGraph transaction simulator")
    parser.add_argument("--count", type=int, default=200, help="Number of transactions to simulate")
    parser.add_argument("--stream", action="store_true", help="Stream continuously")
    parser.add_argument("--interval", type=float, default=0.5, help="Stream interval in seconds")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--dataset", default="mixed",
                        choices=["mixed", "live", "ieee_cis", "paysim", "elliptic"])
    parser.add_argument("--api", default="http://localhost:8000/api")
    args = parser.parse_args()

    API_URL = args.api

    if args.stream:
        asyncio.run(simulate_stream(args.interval, args.dataset))
    else:
        asyncio.run(simulate_batch(args.count, args.concurrency, args.dataset))
