-- Kuzu graph schema for Overfit
-- Compatible with KuzuDB v0.9.0
-- https://kuzudb.com/

-- ---------- Node Tables ----------
-- Problem node table
CREATE NODE TABLE Problem(
  id STRING PRIMARY KEY,
  text STRING,
  type STRING,
  variables STRING,
  metadata STRING
);

-- Method node table
CREATE NODE TABLE Method(
  id STRING PRIMARY KEY,
  name STRING,
  preconditions STRING,
  metadata STRING
);

-- Solution node table  
CREATE NODE TABLE Solution(
  id STRING PRIMARY KEY,
  problem_id STRING,
  expression STRING,
  verified BOOLEAN,
  metadata STRING
);

-- Concept node table
CREATE NODE TABLE Concept(
  id STRING PRIMARY KEY,
  name STRING,
  definition STRING,
  metadata STRING
);

-- ---------- Relationship Tables ----------
-- Link a Solution to the Problem it solves
CREATE REL TABLE SOLVES(
  FROM Solution TO Problem,
  created_at TIMESTAMP
);

-- Methods that can be applied to solve a Problem
CREATE REL TABLE APPLIES_TO(
  FROM Method TO Problem
);

-- Problems are about Concepts
CREATE REL TABLE ABOUT(
  FROM Problem TO Concept
);

-- Methods use Concepts
CREATE REL TABLE USES(
  FROM Method TO Concept
);

-- Solutions reference Concepts (e.g., in the expression/derivation)
CREATE REL TABLE REFERENCES(
  FROM Solution TO Concept
);
