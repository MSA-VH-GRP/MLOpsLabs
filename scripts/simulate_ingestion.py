import argparse
import json
import logging
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone

import urllib.request
from confluent_kafka import Producer

# Assume the project root is in python path, or run as `python -m scripts.simulate_ingestion`
try:
    from src.core.config import settings
    from src.pipelines.topics import NEW_USER, NEW_MOVIE, NEW_RATING
except ImportError:
    # Fallback if run directly without correct PYTHONPATH
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.core.config import settings
    from src.pipelines.topics import NEW_USER, NEW_MOVIE, NEW_RATING


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("simulate_ingestion")

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m")
ZIP_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m.zip")


def download_and_extract_dataset():
    """Download ml-1m.zip if it doesn't exist and extract it."""
    os.makedirs(os.path.dirname(ZIP_FILE), exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        if not os.path.exists(ZIP_FILE):
            logger.info(f"Downloading dataset from {DATASET_URL}...")
            urllib.request.urlretrieve(DATASET_URL, ZIP_FILE)
            logger.info("Download completed.")
        
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(ZIP_FILE))
        logger.info("Extraction completed.")
    else:
        logger.info("Dataset already exists.")


def create_producer():
    """Create a Kafka Producer instance."""
    conf = {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "client.id": "simulated-ingestion-worker"
    }
    return Producer(conf)


def delivery_report(err, msg):
    """Callback for delivery reports from Kafka."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")


def ingest_users(producer: Producer):
    """Parse users.dat and push to NEW_USER topic."""
    users_file = os.path.join(DATA_DIR, "users.dat")
    logger.info("Starting ingestion of users...")
    count = 0
    with open(users_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 5:
                continue
            
            user_data = {
                "event_type": "new_user",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": int(parts[0]),
                "gender": parts[1],
                "age": int(parts[2]),
                "occupation": int(parts[3]),
                "zip_code": parts[4]
            }
            producer.produce(
                NEW_USER,
                key=str(user_data["user_id"]).encode("utf-8"),
                value=json.dumps(user_data).encode("utf-8"),
                callback=delivery_report
            )
            count += 1
            if count % 1000 == 0:
                producer.poll(0)
    producer.flush()
    logger.info(f"Finished ingesting {count} users.")


def ingest_movies(producer: Producer):
    """Parse movies.dat and push to NEW_MOVIE topic."""
    movies_file = os.path.join(DATA_DIR, "movies.dat")
    logger.info("Starting ingestion of movies...")
    count = 0
    with open(movies_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 3:
                continue
            
            movie_data = {
                "event_type": "new_movie",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "movie_id": int(parts[0]),
                "title": parts[1],
                "genres": parts[2].split("|")
            }
            producer.produce(
                NEW_MOVIE,
                key=str(movie_data["movie_id"]).encode("utf-8"),
                value=json.dumps(movie_data).encode("utf-8"),
                callback=delivery_report
            )
            count += 1
            if count % 1000 == 0:
                producer.poll(0)
    producer.flush()
    logger.info(f"Finished ingesting {count} movies.")


def ingest_ratings(producer: Producer, speedup: float):
    """Parse ratings.dat, sort by timestamp, and stream to NEW_RATING topic."""
    ratings_file = os.path.join(DATA_DIR, "ratings.dat")
    logger.info("Loading and sorting ratings...")
    
    ratings = []
    with open(ratings_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            ratings.append({
                "event_type": "new_rating",
                "user_id": int(parts[0]),
                "movie_id": int(parts[1]),
                "rating": int(parts[2]),
                "rating_timestamp": int(parts[3])
            })
            
    # Sort ratings chronologically by the original timestamp
    ratings.sort(key=lambda x: x["rating_timestamp"])
    
    if not ratings:
        logger.info("No ratings found.")
        return

    logger.info(f"Starting real-time playback of {len(ratings)} ratings (speedup={speedup}x)...")
    
    first_rating_time = ratings[0]["rating_timestamp"]
    simulation_start_time = time.time()
    
    count = 0
    for r in ratings:
        # Calculate how much time should have elapsed in the simulation
        simulated_elapsed = (r["rating_timestamp"] - first_rating_time) / speedup
        actual_elapsed = time.time() - simulation_start_time
        
        # If we arrived too early, wait.
        if actual_elapsed < simulated_elapsed:
            time.sleep(simulated_elapsed - actual_elapsed)
            
        r["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        producer.produce(
            NEW_RATING,
            key=f"{r['user_id']}-{r['movie_id']}".encode("utf-8"),
            value=json.dumps(r).encode("utf-8"),
            callback=delivery_report
        )
        count += 1
        if count % 1000 == 0:
            producer.poll(0)
            logger.info(f"Sent {count} ratings...")
            
    producer.flush()
    logger.info("Finished streaming ratings.")


def main():
    parser = argparse.ArgumentParser(description="Simulate MovieLens 1M ingestion to Kafka")
    parser.add_argument("--speedup", type=float, default=1.0, 
                        help="Speedup multiplier for real-time rating ingestion. Higher is faster.")
    parser.add_argument("--skip-users", action="store_true", help="Skip sending users")
    parser.add_argument("--skip-movies", action="store_true", help="Skip sending movies")
    parser.add_argument("--skip-ratings", action="store_true", help="Skip streaming ratings")
    
    args = parser.parse_args()

    download_and_extract_dataset()
    
    producer = create_producer()
    
    if not args.skip_users:
        ingest_users(producer)
        
    if not args.skip_movies:
        ingest_movies(producer)
        
    if not args.skip_ratings:
        ingest_ratings(producer, speedup=args.speedup)


if __name__ == "__main__":
    main()
