
// GDPR DPA Decisions Graph Import Script
// Run in Neo4j Browser or cypher-shell

// Clear existing data (optional)
// MATCH (n) DETACH DELETE n;

// Create constraints
CREATE CONSTRAINT dpa_name IF NOT EXISTS FOR (d:DPA) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE;
CREATE CONSTRAINT principle_name IF NOT EXISTS FOR (p:LegalConcept) REQUIRE p.name IS UNIQUE;

// Load DPA nodes
LOAD CSV WITH HEADERS FROM 'file:///dpa_nodes.csv' AS row
CREATE (d:DPA {
    name: row.name,
    country: row.country,
    country_group: row.country_group
});

// Load Decision nodes
LOAD CSV WITH HEADERS FROM 'file:///decision_nodes.csv' AS row
CREATE (d:Decision {
    decision_id: row.decision_id,
    year: toInteger(row.year),
    fine_eur: toFloat(row.fine_eur),
    breach_case: toBoolean(row.breach_case)
});

// Load Legal Concept nodes
LOAD CSV WITH HEADERS FROM 'file:///principle_nodes.csv' AS row
CREATE (p:LegalConcept {
    name: row.name,
    category: row.category
});

// Load DPA -> Decision relationships
LOAD CSV WITH HEADERS FROM 'file:///dpa_decision_relationships.csv' AS row
MATCH (dpa:DPA {name: split(row.startNodeId, '_')[1]})
MATCH (decision:Decision {decision_id: split(row.endNodeId, '_')[1]})
CREATE (dpa)-[:DECIDED {
    decision_date: row.decision_date,
    fine_amount: toFloat(row.fine_amount)
}]->(decision);

// Load Decision -> Principle relationships
LOAD CSV WITH HEADERS FROM 'file:///decision_principle_relationships.csv' AS row
MATCH (decision:Decision {decision_id: split(row.startNodeId, '_')[1]})
MATCH (principle:LegalConcept {name: split(row.endNodeId, '_')[1]})
CREATE (decision)-[:VIOLATED {category: row.category}]->(principle);

// Create indexes for performance
CREATE INDEX dpa_country IF NOT EXISTS FOR (d:DPA) ON (d.country);
CREATE INDEX decision_year IF NOT EXISTS FOR (d:Decision) ON (d.year);
CREATE INDEX principle_category IF NOT EXISTS FOR (p:LegalConcept) ON (p.category);
