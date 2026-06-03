# Synthetic RAG Test Pack for Saturday.ai

This pack is intentionally synthetic. It is not meant to augment your agent with useful production knowledge. It is designed to test whether your RAG pipeline can retrieve, cite, and synthesize controlled facts.

## Files

- `sat_ai_synthetic_handbook.pdf`: multi-page PDF with versioned rules, planted facts, tables, and sections.
- `sat_ai_synthetic_handbook.txt`: plain-text equivalent for easier baseline ingestion.
- `router_config_v0_1_deprecated.txt`: intentionally conflicting old config.
- `tool_sandbox_policy.txt`: short current policy file for cross-document retrieval.
- `dashboard_event_schema.json`: structured schema file.
- `rag_eval_questions.jsonl`: test questions with expected answer substrings.

## Suggested first experiment

1. Ingest only `sat_ai_synthetic_handbook.txt`.
2. Run the questions in `rag_eval_questions.jsonl`.
3. Ingest the deprecated router file and rerun conflict questions q02 and q07.
4. Ingest the PDF version and compare retrieval quality against the text version.

## What to watch for

- Does the retriever find exact values like `4173`, `420 tokens`, and `90 seconds`?
- Does the answer composer distinguish current rules from deprecated rules?
- Does metadata preserve `document_version`, `section_title`, and `authority_level`?
- Does the system cite the correct file and section?
- Does the agent avoid answering from model memory when the answer is planted in the corpus?
