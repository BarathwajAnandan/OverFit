// Kuzu graph schema for Overfit
// Compatible with KuzuDB Cypher (v0.11.x)
// https://kuzudb.com/

// ---------- Node Tables ----------
CREATE NODE TABLE Problem(
  id STRING,
  text STRING,
  type STRING,
  variables LIST<STRING>,
  metadata STRING,
  PRIMARY KEY (id)
);

CREATE NODE TABLE Method(
  id STRING,
  name STRING,
  preconditions LIST<STRING>,
  metadata STRING,
  PRIMARY KEY (id)
);

CREATE NODE TABLE Solution(
  id STRING,
  problem_id STRING,
  expression STRING,
  verified BOOL,
  metadata STRING,
  PRIMARY KEY (id)
);

CREATE NODE TABLE Concept(
  id STRING,
  name STRING,
  definition STRING,
  metadata STRING,
  PRIMARY KEY (id)
);

// ---------- Relationship Tables ----------
// Link a Solution to the Problem it solves
CREATE REL TABLE SOLVES(
  FROM Solution TO Problem,
  created_at TIMESTAMP
);

// Methods that can be applied to solve a Problem
CREATE REL TABLE APPLIES_TO(
  FROM Method TO Problem
);

// Problems are about Concepts
CREATE REL TABLE ABOUT(
  FROM Problem TO Concept
);

// Methods use Concepts
CREATE REL TABLE USES(
  FROM Method TO Concept
);

// Solutions reference Concepts (e.g., in the expression/derivation)
CREATE REL TABLE REFERENCES(
  FROM Solution TO Concept
); 