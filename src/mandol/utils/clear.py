import logging
import os
import shutil
from neo4j import GraphDatabase
from pymilvus import connections, utility, Collection

# Avoid mutating LogRecord fields before other handlers process the record.
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def clear_neo4j(uri="bolt://localhost:7687", user="neo4j", password="neo4j", database="academicgraph"):
    """Remove neo4j."""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            result = session.run("MATCH (n:MemoryUnit) DETACH DELETE n")
            summary = result.consume()
            
            logging.info(f"Cleared {summary.counters.nodes_deleted} nodes and {summary.counters.relationships_deleted} relationships from Neo4j")
            
            
        driver.close()
        return True
    except Exception as e:
        logging.error(f"Failed to clear Neo4j data: {e}")
        return False

def clear_milvus(host="localhost", port="19530", collection_names=["hippo_memory_units", "my_memory_units"]):
    """Remove milvus."""
    try:
        connections.connect("default", host=host, port=port)
        
        for collection_name in collection_names:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logging.info(f"Deleted Milvus collection: {collection_name}")
                
                # collection = Collection(collection_name)
                # collection.drop()
                
        connections.disconnect("default")
        return True
    except Exception as e:
        logging.error(f"Failed to clear Milvus data: {e}")
        return False

def clear_local_files(save_dir="hippo_save_data"):
    """Remove local files."""
    try:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            logging.info(f"Deleted local data directory: {save_dir}")
        else:
            logging.info(f"Local data directory does not exist: {save_dir}")
        return True
    except Exception as e:
        logging.error(f"Failed to delete local data files: {e}")
        return False

if __name__ == "__main__":
    logging.info("Starting test-data cleanup...")
    
    neo4j_success = clear_neo4j()
    
    milvus_success = clear_milvus()
    
    files_success = clear_local_files()
    
    if neo4j_success and milvus_success and files_success:
        logging.info("All test data cleared successfully")
    else:
        logging.warning("Some test data could not be cleared; check the logs")
