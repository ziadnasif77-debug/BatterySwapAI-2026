"""
generate_dummy_data.py

Generates realistic dummy data for Norwegian IoT battery sensors.
Produces three CSV files in data/raw/:
  - sensor_readings.csv  : per-day voltage/temperature readings per sensor
  - buildings.csv        : 50 Norwegian buildings across 6 cities
  - travel_times.csv     : full travel-time matrix between all buildings
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import math
import os

RNG = np.random.default_rng(42)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# 1. BUILDINGS
# ──────────────────────────────────────────────

CITIES = {
    "Oslo":         {"n": 20, "lat": (59.8, 60.0), "lon": (10.6, 10.9)},
    "Bergen":       {"n": 10, "lat": (60.3, 60.5), "lon": (5.2,  5.4)},
    "Trondheim":    {"n": 10, "lat": (63.3, 63.5), "lon": (10.3, 10.5)},
    "Stavanger":    {"n": 10, "lat": (58.9, 59.1), "lon": (5.6,  5.8)},
    "Tromsø":       {"n":  5, "lat": (69.6, 69.7), "lon": (18.9, 19.1)},
    "Kristiansand": {"n":  5, "lat": (58.1, 58.2), "lon": (7.9,  8.1)},
}

BUILDING_TYPES = ["office", "hospital", "warehouse"]

BUILDING_NAME_PARTS = {
    "Oslo": [
        "Sentrum", "Øst", "Vest", "Nord", "Syd", "Aker", "Grünerløkka", "Frogner",
        "Majorstuen", "Sagene", "Bjørvika", "Holmenkollen", "Ullevål", "Tøyen",
        "Torshov", "Ryen", "Lambertseter", "Grorud", "Stovner", "Furuset",
    ],
    "Bergen": [
        "Bryggen", "Sandviken", "Laksevåg", "Åsane", "Fyllingsdalen",
        "Ytrebygda", "Bergenhus", "Fana", "Arna", "Loddefjord",
    ],
    "Trondheim": [
        "Midtbyen", "Nidarvoll", "Lade", "Heimdal", "Lerkendal",
        "Tiller", "Ranheim", "Saupstad", "Byåsen", "Singsaker",
    ],
    "Stavanger": [
        "Våland", "Hillevåg", "Storhaug", "Eiganes", "Madla",
        "Hundvåg", "Hinna", "Tasta", "Sentrum", "Mariero",
    ],
    "Tromsø": ["Sentrum", "Tromsøya", "Tromsdalen", "Langnes", "Mortensnes"],
    "Kristiansand": ["Sentrum", "Lund", "Kvadraturen", "Vågsbygd", "Randesund"],
}


def generate_buildings() -> pd.DataFrame:
    rows = []
    bid = 1
    for city, cfg in CITIES.items():
        names = BUILDING_NAME_PARTS[city]
        for i in range(cfg["n"]):
            lat   = RNG.uniform(*cfg["lat"])
            lon   = RNG.uniform(*cfg["lon"])
            btype = RNG.choice(BUILDING_TYPES)
            rows.append({
                "building_id":   f"B{bid:03d}",
                "building_name": f"{city} {names[i]}",
                "latitude":      round(float(lat), 6),
                "longitude":     round(float(lon), 6),
                "building_type": btype,
                "city":          city,
            })
            bid += 1
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 2. TRAVEL TIMES
# ──────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def generate_travel_times(buildings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    city_of = dict(zip(buildings["building_id"], buildings["city"]))
    coords  = {row.building_id: (row.latitude, row.longitude)
               for row in buildings.itertuples()}

    for b1 in buildings["building_id"]:
        for b2 in buildings["building_id"]:
            if b1 == b2:
                rows.append({"from_building_id": b1, "to_building_id": b2,
                             "travel_minutes": 0})
                continue
            lat1, lon1 = coords[b1]
            lat2, lon2 = coords[b2]
            dist_km = haversine_km(lat1, lon1, lat2, lon2)
            if city_of[b1] == city_of[b2]:
                base_minutes = dist_km * 3.0
            else:
                base_minutes = dist_km * 1.2 + 30.0
            noise = RNG.uniform(0.9, 1.1)
            rows.append({
                "from_building_id": b1,
                "to_building_id":   b2,
                "travel_minutes":   round(base_minutes * noise, 1),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. SENSOR READINGS
# ──────────────────────────────────────────────

N_SENSORS    = 500
N_DEAD       = int(N_SENSORS * 0.30)   # 150 sensors die
DEAD_THRESH  = 2.5
WINTER_MONTHS = {12, 1, 2, 3}

# Monthly temperature baselines for Norway
MONTHLY_TEMP = {1:-5, 2:-5, 3:-3, 4:5, 5:11, 6:16, 7:18, 8:18,
                9:13, 10:7, 11:2, 12:-3}


def norwegian_temperature(d: date) -> float:
    base  = MONTHLY_TEMP[d.month]
    noise = float(RNG.normal(0, 3))
    return round(base + noise, 2)


def decay_rate_to_die_at(start_v: float, die_at_day: int, dead_thresh: float) -> float:
    """Return decay rate r such that V(die_at_day) = dead_thresh exactly."""
    # V(t) = start_v * exp(-r * t)  ->  r = ln(start_v / dead_thresh) / die_at_day
    return math.log(start_v / dead_thresh) / die_at_day


def generate_sensor_readings(buildings: pd.DataFrame) -> pd.DataFrame:
    building_ids = buildings["building_id"].tolist()
    all_rows = []

    for idx in range(N_SENSORS):
        sensor_id   = f"SEN_{idx + 1:03d}"
        building_id = building_ids[idx % len(building_ids)]
        will_die    = idx < N_DEAD

        # Random start date Jan-Jun 2025
        start_date = date(2025, 1, 1) + timedelta(days=int(RNG.integers(0, 181)))
        n_days     = int(RNG.integers(60, 366))
        start_v    = float(RNG.uniform(3.8, 4.2))

        if will_die:
            # Die somewhere between 60% and 90% into the observation window
            die_at_day  = int(n_days * RNG.uniform(0.60, 0.90))
            die_at_day  = max(die_at_day, 10)
            effective_days = die_at_day * 1.05
            base_rate = decay_rate_to_die_at(start_v, effective_days, DEAD_THRESH)
        else:
            # Living sensor: slow rate that keeps voltage above 2.6V
            max_safe_rate = decay_rate_to_die_at(start_v, n_days * 1.5, DEAD_THRESH)
            base_rate = float(RNG.uniform(max_safe_rate * 0.2, max_safe_rate * 0.8))

        died_on = None

        for d in range(n_days):
            current_date = start_date + timedelta(days=d)
            temp         = norwegian_temperature(current_date)
            is_winter    = current_date.month in WINTER_MONTHS

            rate = base_rate * (1.15 if is_winter else 1.0)
            v    = start_v * math.exp(-rate * d) + float(RNG.normal(0, 0.02))
            v    = round(max(v, 0.0), 4)

            if will_die and died_on is None and v <= DEAD_THRESH:
                died_on = current_date
                all_rows.append({
                    "sensor_id":        sensor_id,
                    "building_id":      building_id,
                    "timestamp":        current_date.isoformat(),
                    "voltage":          v,
                    "temperature":      temp,
                    "end_of_life_date": died_on.isoformat(),
                })
                break

            all_rows.append({
                "sensor_id":        sensor_id,
                "building_id":      building_id,
                "timestamp":        current_date.isoformat(),
                "voltage":          v,
                "temperature":      temp,
                "end_of_life_date": np.nan,
            })

    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

def main():
    print("Generating buildings...")
    buildings     = generate_buildings()
    buildings_out = buildings.drop(columns=["city"])
    buildings_out.to_csv(os.path.join(OUTPUT_DIR, "buildings.csv"), index=False)

    print("Generating travel times...")
    travel = generate_travel_times(buildings)
    travel.to_csv(os.path.join(OUTPUT_DIR, "travel_times.csv"), index=False)

    print("Generating sensor readings...")
    readings = generate_sensor_readings(buildings)
    readings.to_csv(os.path.join(OUTPUT_DIR, "sensor_readings.csv"), index=False)

    total_sensors  = readings["sensor_id"].nunique()
    total_readings = len(readings)
    dead_sensors   = readings[readings["end_of_life_date"].notna()]["sensor_id"].nunique()
    date_min       = readings["timestamp"].min()
    date_max       = readings["timestamp"].max()
    city_counts    = buildings.groupby("city")["building_id"].count()

    print("\n" + "=" * 45)
    print("        DATA GENERATION SUMMARY")
    print("=" * 45)
    print(f"  Total sensors     : {total_sensors}")
    print(f"  Total readings    : {total_readings:,}")
    print(f"  Sensors dead      : {dead_sensors}  ({dead_sensors/total_sensors*100:.0f}%)")
    print(f"  Date range        : {date_min} -> {date_max}")
    print(f"  Buildings per city:")
    for city, count in city_counts.items():
        print(f"    {city:<12}: {count}")
    print("=" * 45)
    print(f"\n  Files saved to: {OUTPUT_DIR}/")
    print("  v sensor_readings.csv")
    print("  v buildings.csv")
    print("  v travel_times.csv")


if __name__ == "__main__":
    main()
