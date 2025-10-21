from neo4j import GraphDatabase

class Neo4jCRUD:
    def __init__(self, uri, user, password, db_name):
        self._uri = uri
        self._user = user
        self._password = password
        self._db_name = db_name
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")

    def close(self):
        if self._driver is not None:
            self._driver.close()
            print("Neo4j connection closed.")

    def create_node(self, label, properties):
        with self._driver.session(database=self._db_name) as session:
            result = session.execute_write(self._create_node, label, properties)
            print(f"Created node: {result}")

    @staticmethod
    def _create_node(tx, label, properties):
        query = (
            f"CREATE (n:{label} $props) "
            "RETURN n"
        )
        result = tx.run(query, props=properties)
        return result.single()[0]

    def find_node(self, label, properties):
        with self._driver.session(database=self._db_name) as session:
            result = session.execute_read(self._find_node, label, properties)
            if result:
                for record in result:
                    print(f"Found node: {record['n']}")
            else:
                print("Node not found.")

    @staticmethod
    def _find_node(tx, label, properties):
        query = f"MATCH (n:{label}) WHERE "
        query += " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query += " RETURN n"
        result = tx.run(query, **properties)
        return result.records()

    def update_node(self, label, properties, new_properties):
        with self._driver.session(database=self._db_name) as session:
            session.execute_write(self._update_node, label, properties, new_properties)
            print("Node updated.")

    @staticmethod
    def _update_node(tx, label, properties, new_properties):
        query = f"MATCH (n:{label}) WHERE "
        query += " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query += " SET "
        query += ", ".join([f"n.{key} = $new_{key}" for key in new_properties.keys()])
        
        params = properties.copy()
        for key, value in new_properties.items():
            params[f"new_{key}"] = value
            
        tx.run(query, **params)

    def delete_node(self, label, properties):
        with self._driver.session(database=self._db_name) as session:
            session.execute_write(self._delete_node, label, properties)
            print("Node deleted.")

    @staticmethod
    def _delete_node(tx, label, properties):
        query = f"MATCH (n:{label}) WHERE "
        query += " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query += " DETACH DELETE n"
        tx.run(query, **properties)

    def create_relationship(self, start_node_label, start_node_props, end_node_label, end_node_props, relationship_type):
        with self._driver.session(database=self._db_name) as session:
            session.execute_write(self._create_relationship, start_node_label, start_node_props, end_node_label, end_node_props, relationship_type)
            print(f"Created relationship '{relationship_type}'.")
    @staticmethod
    def _relationship(self,start_node_label,start_node_props,end_node_labels,end_node_props,relationship_type):
        with  self._driver.session(database=self._db_name) as session:
            session.execute_write(self._create_relationship,start_node_label,start_node_props,end_node_labels,end_node_props,relationship_type)
            print(f"Created relationship '{relationship_type}'.")
    @staticmethod
    def _create_relationship(tx, start_label, start_props, end_label, end_props, rel_type):
        query = (
            f"MATCH (a:{start_label}), (b:{end_label}) "
            "WHERE "
        )
        query += " AND ".join([f"a.{key} = $start_{key}" for key in start_props.keys()])
        query += " AND "
        query += " AND ".join([f"b.{key} = $end_{key}" for key in end_props.keys()])
        query += f" CREATE (a)-[r:{rel_type}]->(b) RETURN type(r)"

        params = {}
        for key, value in start_props.items():
            params[f"start_{key}"] = value
        for key, value in end_props.items():
            params[f"end_{key}"] = value

        tx.run(query, **params)

    @staticmethod
    def _create(tx,start_label,start_props,end_label,end_props,rel_type):
        query=(
            f"MATCH (a:{start_label}),(b:{end_label})"
            "where "
        )
        query += " AND ".join([f"a.{key}= $start_{key}" for key in start_props.keys()])
        query += " AND "
        query += " AND ".join([f"b.{key} = $end_{key}" for key in end_props.keys()])
        query += f"CREATE (a)-[r:{rel_type}]->(b) RETURN type(r)"
        
        params={}
        for key,value in start_props.items():
            params[f"start_{key}"] = value
        for key,value in end_props.items():
            params[f"end_{key}"] = value

        tx.run(query,**params)

if __name__ == "__main__":
    # --- Connection Details ---
    URI = "bolt://localhost:7687"
    USER = "root"
    PASSWORD = "123456"
    DB_NAME = "neo4j"

    # --- Usage Example ---
    db = Neo4jCRUD(URI, USER, PASSWORD, DB_NAME)

    # 1. Create nodes
    print("--- Creating nodes ---")
    db.create_node("Person", {"name": "Alice", "age": 30})
    db.create_node("Person", {"name": "Bob", "age": 25})
    db.create_node("City", {"name": "New York"})

    # 2. Find nodes
    print("\n--- Finding nodes ---")
    db.find_node("Person", {"name": "Alice"})
    db.find_node("City", {"name": "New York"})

    # 3. Create a relationship
    print("\n--- Creating relationship ---")
    db.create_relationship("Person", {"name": "Alice"}, "City", {"name": "New York"}, "LIVES_IN")

    # 4. Update a node
    print("\n--- Updating a node ---")
    db.update_node("Person", {"name": "Alice"}, {"age": 31})
    db.find_node("Person", {"name": "Alice"})

    # 5. Delete a node
    print("\n--- Deleting a node ---")
    db.delete_node("Person", {"name": "Bob"})
    db.find_node("Person", {"name": "Bob"})

    # Close the connection
    db.close()
