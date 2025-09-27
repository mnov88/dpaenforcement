from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .base import BaseExporter


class GraphExporter(BaseExporter):
    """Export data as graph/network formats for relationship analysis."""

    def __init__(self, wide_csv: Path, long_tables_dir: Optional[Path] = None):
        super().__init__(wide_csv, long_tables_dir)
        if NETWORKX_AVAILABLE:
            self._init_graphs()

    def _init_graphs(self) -> None:
        """Initialize graph structures."""
        self.dpa_decision_graph = nx.Graph()
        self.decision_article_graph = nx.Graph()  # Use regular Graph, mark bipartite with node attributes
        self.violation_network = nx.Graph()
        self.cross_border_graph = nx.DiGraph()

    def export(self, output_dir: Path) -> None:
        """Export various graph representations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build different graph representations
        self._build_dpa_decision_network(output_dir)
        self._build_decision_article_bipartite(output_dir)
        self._build_violation_cooccurrence_network(output_dir)
        self._build_cross_border_network(output_dir)

        # Export in multiple formats
        self._export_graphml(output_dir)
        self._export_networkx_formats(output_dir)
        self._export_neo4j_csv(output_dir)
        self._export_graph_metadata(output_dir)

    def _build_dpa_decision_network(self, output_dir: Path) -> None:
        """Build DPA-Decision network with attributes."""
        df = self.df

        # Add DPA nodes
        dpas = df['dpa_name_canonical'].dropna().unique()
        for dpa in dpas:
            dpa_decisions = df[df['dpa_name_canonical'] == dpa]
            total_fines = float(dpa_decisions['fine_eur'].sum()) if 'fine_eur' in dpa_decisions.columns else 0.0
            avg_fine = float(dpa_decisions['fine_eur'].mean()) if 'fine_eur' in dpa_decisions.columns else 0.0
            breach_cases = int(dpa_decisions['breach_case'].sum()) if 'breach_case' in dpa_decisions.columns else 0

            self.dpa_decision_graph.add_node(
                dpa,
                node_type='dpa',
                country=str(dpa_decisions['country_code'].iloc[0]) if len(dpa_decisions) > 0 else None,
                country_group=str(dpa_decisions['country_group'].iloc[0]) if len(dpa_decisions) > 0 else None,
                total_decisions=int(len(dpa_decisions)),
                total_fines_eur=total_fines,
                avg_fine_eur=avg_fine,
                breach_cases=breach_cases
            )

        # Add Decision nodes and edges
        for _, row in df.iterrows():
            decision_id = row['decision_id']
            dpa = row['dpa_name_canonical']

            if pd.notna(dpa) and pd.notna(decision_id):
                # Add decision node
                decision_year = row.get('decision_year')
                decision_year_int = int(decision_year) if pd.notna(decision_year) else None

                fine_eur = row.get('fine_eur', 0)
                fine_eur_float = float(fine_eur) if pd.notna(fine_eur) else 0.0

                n_principles = row.get('n_principles_violated', 0)
                n_principles_int = int(n_principles) if pd.notna(n_principles) else 0

                # Handle nullable booleans safely
                breach_case_val = row.get('breach_case', False)
                breach_case = None if pd.isna(breach_case_val) else bool(breach_case_val)

                severity_val = row.get('severity_measures_present', False)
                severity_measures = None if pd.isna(severity_val) else bool(severity_val)

                cross_border_val = self._is_cross_border(row)
                cross_border = None if pd.isna(cross_border_val) else bool(cross_border_val)

                self.dpa_decision_graph.add_node(
                    decision_id,
                    node_type='decision',
                    year=decision_year_int,
                    quarter=str(row.get('decision_quarter')) if pd.notna(row.get('decision_quarter')) else None,
                    fine_eur=fine_eur_float,
                    breach_case=breach_case,
                    severity_measures=severity_measures,
                    n_principles_violated=n_principles_int,
                    cross_border=cross_border
                )

                # Add edge between DPA and Decision
                decision_date = row.get('decision_date')
                decision_date_str = str(decision_date) if pd.notna(decision_date) else None

                self.dpa_decision_graph.add_edge(
                    dpa, decision_id,
                    edge_type='decided_by',
                    decision_date=decision_date_str,
                    fine_amount=float(row.get('fine_eur', 0)) if pd.notna(row.get('fine_eur')) else 0.0
                )

    def _build_decision_article_bipartite(self, output_dir: Path) -> None:
        """Build bipartite graph of decisions and violated articles/principles."""
        df = self.df

        # Extract violation relationships from multi-select columns
        violation_columns = {
            'q31_violated': 'Article 5 Principles',
            'q57_rights_violated': 'Data Subject Rights',
            'q53_powers': 'Corrective Powers'
        }

        for _, row in df.iterrows():
            decision_id = row['decision_id']
            if pd.isna(decision_id):
                continue

            # Add decision node
            decision_year = row.get('decision_year')
            decision_year_int = int(decision_year) if pd.notna(decision_year) else None

            fine_eur = row.get('fine_eur', 0)
            fine_eur_float = float(fine_eur) if pd.notna(fine_eur) else 0.0

            self.decision_article_graph.add_node(
                decision_id,
                bipartite=0,  # Decision side
                node_type='decision',
                year=decision_year_int,
                fine_eur=fine_eur_float,
                country=str(row.get('country_code')) if pd.notna(row.get('country_code')) else None,
                dpa=str(row.get('dpa_name_canonical')) if pd.notna(row.get('dpa_name_canonical')) else None
            )

            # Add article/principle nodes and relationships
            for col_prefix, category in violation_columns.items():
                # Get all boolean columns for this category
                bool_cols = [col for col in df.columns if col.startswith(f"{col_prefix}_") and
                           not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]

                for col in bool_cols:
                    col_value = row.get(col)
                    if pd.notna(col_value) and col_value == 1:  # Principle/right is violated/exercised
                        principle = col.replace(f"{col_prefix}_", "")
                        node_id = f"{category}::{principle}"

                        # Add principle node
                        if not self.decision_article_graph.has_node(node_id):
                            self.decision_article_graph.add_node(
                                node_id,
                                bipartite=1,  # Article/principle side
                                node_type='legal_concept',
                                category=category,
                                principle=principle
                            )

                        # Add edge
                        edge_type = 'violated' if 'violated' in col_prefix else 'exercised'
                        self.decision_article_graph.add_edge(
                            decision_id, node_id,
                            edge_type=edge_type,
                            category=category
                        )

    def _build_violation_cooccurrence_network(self, output_dir: Path) -> None:
        """Build network of co-occurring violations."""
        if not self.long_tables_dir:
            return

        try:
            # Load violation data
            violations_df = self.load_long_table('article_5_violated.csv')
            rights_df = self.load_long_table('rights_violated.csv')

            # Combine violation types
            all_violations = pd.concat([
                violations_df[violations_df['token_status'] == 'KNOWN'].assign(type='Article 5'),
                rights_df[rights_df['token_status'] == 'KNOWN'].assign(type='Rights')
            ])

            # Group by decision to find co-occurrences
            decision_groups = all_violations.groupby('decision_id')

            # Build co-occurrence matrix
            violation_pairs = []
            for decision_id, group in decision_groups:
                options = group['option'].tolist()
                if len(options) > 1:
                    # Create pairs of co-occurring violations
                    for i, opt1 in enumerate(options):
                        for opt2 in options[i+1:]:
                            violation_pairs.append((opt1, opt2, decision_id))

            # Build network
            for opt1, opt2, decision_id in violation_pairs:
                if not self.violation_network.has_edge(opt1, opt2):
                    self.violation_network.add_edge(opt1, opt2, weight=0, decisions_count=0)

                # Increment weight and track decisions count
                self.violation_network[opt1][opt2]['weight'] += 1
                self.violation_network[opt1][opt2]['decisions_count'] += 1

            # Add node attributes
            for node in self.violation_network.nodes():
                node_violations = all_violations[all_violations['option'] == node]
                self.violation_network.nodes[node].update({
                    'total_occurrences': int(len(node_violations)),
                    'violation_type': str(node_violations['type'].iloc[0]) if len(node_violations) > 0 else 'Unknown'
                })

        except FileNotFoundError:
            pass  # Skip if long tables not available

    def _build_cross_border_network(self, output_dir: Path) -> None:
        """Build network of cross-border enforcement relationships."""
        df = self.df

        # Filter cross-border cases
        q49_mask = df['raw_q49'].fillna('').str.contains('YES_', na=False) if 'raw_q49' in df.columns else pd.Series(False, index=df.index)
        q62_mask = df['raw_q62'].fillna('').str.contains('LEAD_|CONCERNED_|JOINT_', na=False) if 'raw_q62' in df.columns else pd.Series(False, index=df.index)
        cross_border_cases = df[q49_mask | q62_mask]

        for _, row in cross_border_cases.iterrows():
            decision_id = row['decision_id']
            dpa = row['dpa_name_canonical']
            country = row['country_code']

            if pd.notna(dpa) and pd.notna(country):
                # Add nodes
                if not self.cross_border_graph.has_node(dpa):
                    self.cross_border_graph.add_node(
                        dpa,
                        node_type='dpa',
                        country=country,
                        cross_border_cases=0
                    )

                self.cross_border_graph.nodes[dpa]['cross_border_cases'] += 1

                # For simplification, we'll create edges based on similar case types
                # In practice, this would need more detailed parsing of the cooperation described

    def _is_cross_border(self, row: pd.Series) -> bool:
        """Determine if a case involves cross-border processing."""
        q49 = str(row.get('q49', ''))
        q62 = str(row.get('q62', ''))
        return ('YES_' in q49 or 'LEAD_' in q62 or 'CONCERNED_' in q62 or 'JOINT_' in q62)

    def _export_graphml(self, output_dir: Path) -> None:
        """Export graphs in GraphML format (with GML fallback)."""
        if not NETWORKX_AVAILABLE:
            return

        graphs = {
            'dpa_decision_network': self.dpa_decision_graph,
            'decision_article_bipartite': self.decision_article_graph,
            'violation_cooccurrence': self.violation_network,
            'cross_border_network': self.cross_border_graph
        }

        for name, graph in graphs.items():
            if graph.number_of_nodes() > 0:
                try:
                    # Try GraphML first
                    nx.write_graphml(graph, output_dir / f"{name}.graphml")
                except (TypeError, ValueError) as e:
                    # Fallback to GML
                    print(f"GraphML export failed for {name}, using GML: {e}")
                    try:
                        nx.write_gml(graph, output_dir / f"{name}.gml")
                    except Exception:
                        # Final fallback to edge list
                        self._export_edge_list(graph, output_dir / f"{name}_edges.csv")

    def _export_networkx_formats(self, output_dir: Path) -> None:
        """Export in NetworkX-compatible formats for Python/R."""
        if not NETWORKX_AVAILABLE:
            return

        # Export as GML for R igraph compatibility
        graphs = {
            'dpa_decision_network.gml': self.dpa_decision_graph,
            'decision_article_bipartite.gml': self.decision_article_graph,
            'violation_cooccurrence.gml': self.violation_network
        }

        for filename, graph in graphs.items():
            if graph.number_of_nodes() > 0:
                try:
                    nx.write_gml(graph, output_dir / filename)
                except Exception:
                    # Fallback to edge list if GML fails
                    edge_file = filename.replace('.gml', '_edges.csv')
                    self._export_edge_list(graph, output_dir / edge_file)

        # Export adjacency matrices for analysis
        self._export_adjacency_matrices(output_dir)

    def _export_edge_list(self, graph: nx.Graph, output_path: Path) -> None:
        """Export graph as edge list CSV."""
        edges = []
        for u, v, data in graph.edges(data=True):
            edge_row = {'source': u, 'target': v}
            edge_row.update(data)
            edges.append(edge_row)

        if edges:
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)

    def _export_adjacency_matrices(self, output_dir: Path) -> None:
        """Export adjacency matrices for network analysis."""
        if not NETWORKX_AVAILABLE:
            return

        matrices_dir = output_dir / "adjacency_matrices"
        matrices_dir.mkdir(exist_ok=True)

        graphs = {
            'violation_cooccurrence': self.violation_network,
            'dpa_decisions': self.dpa_decision_graph
        }

        for name, graph in graphs.items():
            if graph.number_of_nodes() > 0:
                # Adjacency matrix
                adj_matrix = nx.adjacency_matrix(graph)
                nodes = list(graph.nodes())

                # Save as CSV with node labels
                df = pd.DataFrame(
                    adj_matrix.toarray(),
                    index=nodes,
                    columns=nodes
                )
                df.to_csv(matrices_dir / f"{name}_adjacency.csv")

                # Node attributes
                node_attrs = []
                for node in nodes:
                    attrs = {'node': node}
                    attrs.update(graph.nodes[node])
                    node_attrs.append(attrs)

                pd.DataFrame(node_attrs).to_csv(
                    matrices_dir / f"{name}_nodes.csv",
                    index=False
                )

    def _export_neo4j_csv(self, output_dir: Path) -> None:
        """Export CSV files for Neo4j import."""
        neo4j_dir = output_dir / "neo4j_import"
        neo4j_dir.mkdir(exist_ok=True)

        # Nodes CSV
        self._export_neo4j_nodes(neo4j_dir)

        # Relationships CSV
        self._export_neo4j_relationships(neo4j_dir)

        # Create import script
        self._create_neo4j_import_script(neo4j_dir)

    def _export_neo4j_nodes(self, neo4j_dir: Path) -> None:
        """Export nodes for Neo4j import."""
        df = self.df

        # DPA nodes
        dpa_nodes = []
        for dpa in df['dpa_name_canonical'].dropna().unique():
            dpa_data = df[df['dpa_name_canonical'] == dpa].iloc[0]
            dpa_nodes.append({
                'nodeId': f"dpa_{dpa.replace(' ', '_')}",
                'name': dpa,
                'type': 'DPA',
                'country': dpa_data.get('country_code'),
                'country_group': dpa_data.get('country_group')
            })

        pd.DataFrame(dpa_nodes).to_csv(neo4j_dir / "dpa_nodes.csv", index=False)

        # Decision nodes
        decision_nodes = []
        for _, row in df.iterrows():
            if pd.notna(row['decision_id']):
                decision_nodes.append({
                    'nodeId': f"decision_{row['decision_id']}",
                    'decision_id': row['decision_id'],
                    'type': 'Decision',
                    'year': row.get('decision_year'),
                    'fine_eur': row.get('fine_eur', 0),
                    'breach_case': row.get('breach_case', False)
                })

        pd.DataFrame(decision_nodes).to_csv(neo4j_dir / "decision_nodes.csv", index=False)

        # Article/Principle nodes
        principle_nodes = []
        legal_concepts = set()

        # Extract from multi-select columns
        multi_prefixes = ['q31_violated', 'q57_rights_violated', 'q53_powers']
        for prefix in multi_prefixes:
            bool_cols = [col for col in df.columns if col.startswith(f"{prefix}_") and
                        not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]
            for col in bool_cols:
                principle = col.replace(f"{prefix}_", "")
                legal_concepts.add((principle, prefix.split('_')[1]))

        for principle, category in legal_concepts:
            principle_nodes.append({
                'nodeId': f"principle_{principle}",
                'name': principle,
                'type': 'LegalConcept',
                'category': category
            })

        pd.DataFrame(principle_nodes).to_csv(neo4j_dir / "principle_nodes.csv", index=False)

    def _export_neo4j_relationships(self, neo4j_dir: Path) -> None:
        """Export relationships for Neo4j import."""
        df = self.df

        # DPA -> Decision relationships
        dpa_decision_rels = []
        for _, row in df.iterrows():
            if pd.notna(row['decision_id']) and pd.notna(row['dpa_name_canonical']):
                dpa_decision_rels.append({
                    'startNodeId': f"dpa_{row['dpa_name_canonical'].replace(' ', '_')}",
                    'endNodeId': f"decision_{row['decision_id']}",
                    'type': 'DECIDED',
                    'decision_date': row.get('decision_date'),
                    'fine_amount': row.get('fine_eur', 0)
                })

        pd.DataFrame(dpa_decision_rels).to_csv(neo4j_dir / "dpa_decision_relationships.csv", index=False)

        # Decision -> Principle relationships
        decision_principle_rels = []
        violation_prefixes = ['q31_violated', 'q57_rights_violated']

        for _, row in df.iterrows():
            if pd.isna(row['decision_id']):
                continue

            for prefix in violation_prefixes:
                bool_cols = [col for col in df.columns if col.startswith(f"{prefix}_") and
                           not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]

                for col in bool_cols:
                    col_value = row.get(col)
                    if pd.notna(col_value) and col_value == 1:
                        principle = col.replace(f"{prefix}_", "")
                        decision_principle_rels.append({
                            'startNodeId': f"decision_{row['decision_id']}",
                            'endNodeId': f"principle_{principle}",
                            'type': 'VIOLATED' if 'violated' in prefix else 'DISCUSSED',
                            'category': prefix.split('_')[1]
                        })

        pd.DataFrame(decision_principle_rels).to_csv(neo4j_dir / "decision_principle_relationships.csv", index=False)

    def _create_neo4j_import_script(self, neo4j_dir: Path) -> None:
        """Create Cypher script for Neo4j import."""
        script = """
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
"""

        with open(neo4j_dir / "import_script.cypher", "w") as f:
            f.write(script)

    def _export_graph_metadata(self, output_dir: Path) -> None:
        """Export metadata about the graph structures."""
        metadata = {
            'graph_statistics': {},
            'formats_available': [
                'GraphML (Gephi, Cytoscape compatible)',
                'GML (R igraph compatible)',
                'CSV edge lists (general purpose)',
                'Neo4j import CSVs (property graph database)',
                'Adjacency matrices (mathematical analysis)'
            ],
            'analysis_suggestions': {
                'dpa_decision_network': [
                    'Analyze DPA enforcement patterns by country/region',
                    'Identify central DPAs in enforcement activity',
                    'Study temporal patterns in decision making'
                ],
                'decision_article_bipartite': [
                    'Bipartite network analysis of violations and decisions',
                    'Identify most commonly violated principles',
                    'Project to one-mode for principle co-occurrence'
                ],
                'violation_cooccurrence': [
                    'Network analysis of co-occurring violations',
                    'Community detection for violation clusters',
                    'Centrality analysis for key principles'
                ],
                'cross_border_network': [
                    'Analysis of international cooperation patterns',
                    'Lead authority relationships',
                    'Geographic enforcement networks'
                ]
            },
            'legal_interpretation_notes': [
                'Edges represent legal relationships, not causal connections',
                'Node centrality indicates enforcement frequency, not legal importance',
                'Co-occurrence does not imply legal dependency',
                'Cross-border edges represent cooperation, not jurisdiction'
            ]
        }

        # Add actual statistics if NetworkX is available
        if NETWORKX_AVAILABLE:
            graphs = {
                'dpa_decision_network': self.dpa_decision_graph,
                'decision_article_bipartite': self.decision_article_graph,
                'violation_cooccurrence': self.violation_network,
                'cross_border_network': self.cross_border_graph
            }

            for name, graph in graphs.items():
                if graph.number_of_nodes() > 0:
                    stats = {
                        'nodes': graph.number_of_nodes(),
                        'edges': graph.number_of_edges(),
                        'density': nx.density(graph) if graph.number_of_nodes() > 1 else 0
                    }

                    if hasattr(graph, 'nodes') and len(graph.nodes()) > 0:
                        # Add centrality measures for smaller graphs
                        if graph.number_of_nodes() < 1000:
                            try:
                                centrality = nx.degree_centrality(graph)
                                stats['max_degree_centrality'] = max(centrality.values())
                                stats['avg_degree_centrality'] = sum(centrality.values()) / len(centrality)
                            except:
                                pass

                    metadata['graph_statistics'][name] = stats

        with open(output_dir / "graph_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)