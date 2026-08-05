# Built-in Graph Template Audit

Source: `/home/sikmindz/Coding/agent-graph-mcp-release/src/templates.rs`

## Count

- **5 available/executable templates** are listed by `list()`.
- **3 unavailable templates** are listed with typed reasons.
- **8 catalog entries total** (5 available + 3 unavailable).
- `research_pipeline` is also accepted by `instantiate()` as a **legacy alias** of `analysis_pipeline`, but is not an additional available template.

## Available templates and graph shapes

1. **`council_deliberation` (v2)**
   - Shape: coordinator → fanout → three parallel analyst branches (`analyst_0`, `analyst_1`, `analyst_2`) → join/collect-array → synthesize → END.
   - Nodes: coordinator, fanout, 3 analysts, join, synthesize (7 nodes).
   - Parallelism: 3.

2. **`parallel_council` (v1)**
   - Shape: fanout → two parallel debate branches (optimist, skeptic) → join/collect-array → judge → END.
   - Nodes: fanout, optimist, skeptic, join, judge (5 nodes).
   - Parallelism: 2.

3. **`plan_critique_refine` (v1)**
   - Shape: linear sequential pipeline plan → critique → refine → END.
   - Nodes: plan, critique, refine (3 nodes).

4. **`analysis_pipeline` (v1)**
   - Shape: planner → researcher → extractor → synthesizer → validator → conditional router.
   - Valid path: router → formatter → END.
   - Invalid path: router → corrector → validator (correction loop), with max-iteration limit.
   - Nodes: planner, researcher, extractor, synthesizer, validator, validation_router, corrector, formatter (8 nodes).

5. **`classifier_router` (v2)**
   - Shape: classifier → conditional router → one of four handlers (`bug_handler`, `feature_handler`, `question_handler`, `general_handler`), each → END.
   - Nodes: classifier, router, 4 handlers (6 nodes).
   - Branching is based on `classification.label`; original input is preserved for handlers.

## Unavailable templates

- **`approval_gated_action`** — requires authenticated human approval/HITL operator authority, not installed/verified.
- **`research_pipeline`** — renamed to `analysis_pipeline`; true web research/source verification is not implemented. Despite being unavailable in the catalog, the legacy ID still instantiates as an alias to `analysis_pipeline`.
- **`map_reduce`** — requires a dynamic parallel branch count from input data.

No source files were modified; report created at `/tmp/subagent-7-report.md`.
