"""Manual connectivity checks for optional database integrations."""

import logging
import argparse
from neo4j import GraphDatabase
from pymilvus import connections, utility

# Avoid mutating LogRecord fields before other handlers process the record.
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def test_neo4j_connection(uri="bolt://localhost:7687", user="neo4j", password="neo4j", database="academicgraph"):
    """Run test neo4j connection."""
    try:
        logging.info(f"Connecting to Neo4j server: {uri}, database: {database}")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session(database=database) as session:
            result = session.run("RETURN 'Neo4j connection succeeded!' AS message")
            message = result.single()[0]
            logging.info(f"Neo4j test result: {message}")
        
        with driver.session(database=database) as session:
            result = session.run("CALL db.info()")
            db_info = result.single().data()
            logging.info(f"Neo4j database info: {db_info}")
        
        driver.close()
        logging.info("Neo4j connection test succeeded")
        return True
    
    except Exception as e:
        logging.error(f"Neo4j connection test failed: {e}")
        return False

def test_milvus_connection(host="localhost", port="19530", user="", password=""):
    """Run test milvus connection."""
    try:
        logging.info(f"Connecting to Milvus server: {host}:{port}")
        connections.connect(
            alias="default", 
            host=host, 
            port=port,
            user=user,
            password=password
        )
        
        if connections.has_connection("default"):
            logging.info("Milvus connection established")
            
            collections = utility.list_collections()
            logging.info(f"Milvus collections: {collections}")
            
            status = utility.get_server_version()
            logging.info(f"Milvus server version: {status}")
            
            connections.disconnect("default")
            logging.info("Milvus connection closed")
            return True
        else:
            logging.error("Milvus connection failed")
            return False
            
    except Exception as e:
        logging.error(f"Milvus connection test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test database connections')
    
    parser.add_argument('--neo4j_uri', default='bolt://localhost:7687', help='Neo4j server URI')
    parser.add_argument('--neo4j_user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', default='neo4j', help='Neo4j password')
    parser.add_argument('--neo4j_database', default='academicgraph', help='Neo4j database name')
    
    parser.add_argument('--milvus_host', default='localhost', help='Milvus server host')
    parser.add_argument('--milvus_port', default='19530', help='Milvus server port')
    parser.add_argument('--milvus_user', default='', help='Milvus username')
    parser.add_argument('--milvus_password', default='', help='Milvus password')
    
    parser.add_argument('--test_neo4j', action='store_true', help='Test the Neo4j connection')
    parser.add_argument('--test_milvus', action='store_true', help='Test the Milvus connection')
    parser.add_argument('--test_all', action='store_true', help='Test all database connections')
    
    args = parser.parse_args()
    
    if not (args.test_neo4j or args.test_milvus):
        args.test_all = True
    
    logging.info("Starting database connection tests...")
    
    if args.test_neo4j or args.test_all:
        neo4j_success = test_neo4j_connection(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        print(f"\nNeo4j connection test: {'succeeded' if neo4j_success else 'failed'}")
    
    if args.test_milvus or args.test_all:
        milvus_success = test_milvus_connection(
            host=args.milvus_host,
            port=args.milvus_port,
            user=args.milvus_user,
            password=args.milvus_password
        )
        print(f"\nMilvus connection test: {'succeeded' if milvus_success else 'failed'}")
    
    logging.info("Database connection tests completed")

if __name__ == "__main__":
    main()
