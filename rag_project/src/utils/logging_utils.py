import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseSettings

def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("rag-system")

class MetricsTracker:
    def __init__(self):
        self.query_count = 0
        self.avg_latency = 0
        self.queries = []
    
    def track_query(self, query, latency, num_chunks_retrieved):
        self.query_count += 1
        
        # Update average latency using running average
        self.avg_latency = (self.avg_latency * (self.query_count - 1) + latency) / self.query_count
        
        # Store query details
        self.queries.append({
            "query": query,
            "latency": latency,
            "chunks_retrieved": num_chunks_retrieved,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_metrics(self):
        return {
            "total_queries": self.query_count,
            "avg_latency_seconds": self.avg_latency,
            "recent_queries": self.queries[-10:] if self.queries else []
        }