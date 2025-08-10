CREATE NODE TABLE Problem(
  id STRING PRIMARY KEY,
  text STRING,
  type STRING,
  variables STRING,
  metadata STRING
);

CREATE NODE TABLE Method(
  id STRING PRIMARY KEY,
  name STRING,
  preconditions STRING,
  metadata STRING
);

CREATE NODE TABLE Solution(
  id STRING PRIMARY KEY,
  problem_id STRING,
  expression STRING,
  verified BOOLEAN,
  metadata STRING
);

CREATE NODE TABLE Concept(
  id STRING PRIMARY KEY,
  name STRING,
  definition STRING,
  metadata STRING
);

CREATE REL TABLE SOLVES(
  FROM Solution TO Problem,
  created_at TIMESTAMP
);

CREATE REL TABLE APPLIES_TO(
  FROM Method TO Problem
);

CREATE REL TABLE ABOUT(
  FROM Problem TO Concept
);

CREATE REL TABLE USES(
  FROM Method TO Concept
);

CREATE REL TABLE REFERENCES(
  FROM Solution TO Concept
);

CREATE REL TABLE RELATED_TO(FROM Problem TO Problem, similarity FLOAT);
CREATE REL TABLE DEPENDS_ON(FROM Concept TO Concept);
CREATE REL TABLE SIMILAR_TO(FROM Solution TO Solution, similarity FLOAT);
CREATE REL TABLE CAUSED_BY(FROM Problem TO Problem);
CREATE REL TABLE WORKAROUND_FOR(FROM Method TO Problem);

CREATE NODE TABLE Model(
  id STRING PRIMARY KEY,
  name STRING,
  metadata STRING
);

CREATE (:Model {id: 'gpt2', name: 'GPT-2', metadata: '{}'});
CREATE (:Model {id: 'gpt2_124m', name: 'GPT-2 124M', metadata: '{}'});
CREATE (:Model {id: 'qwen_14b', name: 'Qwen-14B-Chat', metadata: '{}'});
CREATE (:Model {id: 'deepseek_r1', name: 'DeepSeek R1', metadata: '{}'});
CREATE (:Model {id: 'llama2', name: 'Llama 2', metadata: '{}'});
CREATE (:Model {id: 'claude_opus', name: 'Claude Opus', metadata: '{}'});
CREATE (:Model {id: 'gpt4', name: 'GPT-4', metadata: '{}'});

CREATE REL TABLE CAN_SOLVE(FROM Model TO Problem, confidence FLOAT);
CREATE REL TABLE CANT_SOLVE(FROM Model TO Problem, reason STRING);
CREATE REL TABLE RELATED_TO(FROM Problem TO Problem, similarity FLOAT);
CREATE REL TABLE DEPENDS_ON(FROM Concept TO Concept);
CREATE REL TABLE SIMILAR_TO(FROM Solution TO Solution, similarity FLOAT);
CREATE REL TABLE CAUSED_BY(FROM Problem TO Problem);
CREATE REL TABLE WORKAROUND_FOR(FROM Method TO Problem);
