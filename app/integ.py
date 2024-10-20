from cassandra.cluster import Cluster

# Initialize Cassandra cluster connection
def cassandra_connect():
    cluster = Cluster(['127.0.0.1'])  # Replace with your actual Cassandra node address
    session = cluster.connect('waste_management')  # Replace with your keyspace
    return session
