# Repository Restructure Log

This document tracks the restructuring of the repository. It serves as a reference for:
1. The original directory structure (before any moves)
2. A log of every move performed (old path → new path)
3. A checklist for updating inter-repo links after all moves are complete

---

## Phase 1: Current Directory Structure (Before Restructuring)

```
/
├── .gitattributes
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── evals workshop.png
│
├── 01-operational-metrics/
│   ├── .gitkeep
│   ├── 01-Operational-Metrics.ipynb
│   ├── README.md
│   ├── data/
│   │   └── emails/
│   │       ├── sampleemail1.txt
│   │       └── sampleemail2.txt
│   └── images/
│       ├── Custom-operational-metrics-cloudwatch-dashboard.png
│       └── operational-metrics-cloudwatch-dashboard.png
│
├── 02-quality-metrics/
│   ├── .gitkeep
│   ├── 01_LLM_as_Judge_analysis.ipynb
│   ├── 02_LLM_as_Jury_evaluation_analysis.ipynb
│   ├── 03_Evaluating_your_Judge.ipynb
│   ├── README.md
│   ├── city_pop.csv
│   ├── judge_benchmark.jsonl
│   └── utils.py
│
├── 03-agentic-metrics/
│   ├── 03-Agentic-Metrics.ipynb
│   ├── README.md
│   ├── city_pop.csv
│   ├── config.yaml
│   ├── core.txt
│   ├── data.csv
│   ├── dataset.json
│   ├── evaluation_results.json
│   ├── hello.txt
│   ├── log.txt
│   ├── notes.txt
│   ├── status.txt
│   ├── test_cases.json
│   ├── todo.md
│   ├── data/
│   │   ├── labeled_traces.json
│   │   ├── raw_traces.json
│   │   └── synthetic_queries_for_analysis.csv
│   └── images/
│       └── architecture.png
│
├── 04-workload-specific-evaluations/
│   ├── 04-01-Intelligent-Document-Processing/
│   │   ├── .gitkeep
│   │   ├── 04-01-Simple-structured-data-evaluation.ipynb
│   │   └── readme.md
│   │
│   ├── 04-02-Guardrails/
│   │   ├── 04-02-01-filters.ipynb
│   │   ├── 04-02-02-grounding.ipynb
│   │   ├── 04-02-03-alignment.ipynb
│   │   ├── 04-02-04-operational.ipynb
│   │   ├── 04-02-05-reasoning.md
│   │   ├── 04-02-06-evaluation.ipynb
│   │   ├── README.md
│   │   ├── calendar.pdf
│   │   ├── housing-code.pdf
│   │   ├── requirements.txt
│   │   └── data/
│   │       ├── tests.json
│   │       └── tests_with_adversarial.json
│   │
│   ├── 04-03-Basic-RAG/
│   │   ├── README.md
│   │   ├── data/
│   │   │   └── eval-datasets/
│   │   │       ├── 1_embeddings_validation.csv
│   │   │       └── 5_e2e_validation.csv
│   │   ├── example-notebook/
│   │   │   ├── 04-03-Basic-RAG-Evaluation.ipynb
│   │   │   └── requirements.txt
│   │   └── images/
│   │       └── knowledge_store_overview.png
│   │
│   ├── 04-04-MultiModal-RAG/
│   │   ├── .gitkeep
│   │   ├── 04-04-01-Multimodal-RAG.ipynb
│   │   ├── readme.md
│   │   ├── requirements.txt
│   │   └── utils.py
│   │
│   ├── 04-05-Speech-to-Speech/
│   │   ├── .env.example
│   │   ├── .env.test
│   │   ├── .gitignore
│   │   ├── README.md
│   │   ├── auto_classifier.py
│   │   ├── env.example
│   │   ├── environment.py
│   │   ├── package.json
│   │   ├── playwright.config.ts
│   │   ├── requirements.txt
│   │   ├── s2s_annotation_ui.py
│   │   ├── s2s_entire_eval_pipeline.ipynb
│   │   ├── s2s_evaluator.py
│   │   ├── sagemaker_helper.py
│   │   ├── setup.sh
│   │   ├── assets/
│   │   │   └── images/
│   │   │       ├── EvaluationVisualizations.png
│   │   │       ├── PlayWrightTest.png
│   │   │       ├── TestWorkbenchPlayground.png
│   │   │       ├── TestWorkbenchSettings.png
│   │   │       ├── architecture.excalidraw
│   │   │       └── architecture.png
│   │   ├── config/
│   │   │   ├── llm_judge_s2s_config.json
│   │   │   └── manual_mappings.json
│   │   ├── data/
│   │   │   ├── s2s_eval_data.jsonl
│   │   │   └── s2s_validation_dataset.jsonl
│   │   ├── sample_s2s_app/
│   │   │   ├── README.md
│   │   │   ├── python-server/
│   │   │   │   ├── audio_buffer.py
│   │   │   │   ├── conversation_history.py
│   │   │   │   ├── env.example
│   │   │   │   ├── nova_sonic_pricing.py
│   │   │   │   ├── opentelemetry_span_manager.py
│   │   │   │   ├── requirements.txt
│   │   │   │   ├── run_aiohttp_server.py
│   │   │   │   ├── run_server_with_telemetry.sh
│   │   │   │   ├── s2s_events.py
│   │   │   │   ├── s2s_session_manager.py
│   │   │   │   ├── server.py
│   │   │   │   ├── session_config.json
│   │   │   │   ├── session_info.py
│   │   │   │   ├── session_state.py
│   │   │   │   ├── session_transition_manager.py
│   │   │   │   ├── tool_registry.py
│   │   │   │   ├── integration/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── strands_agent.py
│   │   │   │   └── recordings/
│   │   │   │       └── .gitignore
│   │   │   └── react-client/
│   │   │       ├── .env
│   │   │       ├── package.json
│   │   │       ├── public/
│   │   │       │   ├── favicon.ico
│   │   │       │   ├── index.html
│   │   │       │   ├── manifest.json
│   │   │       │   └── robots.txt
│   │   │       └── src/
│   │   │           ├── App.js
│   │   │           ├── index.css
│   │   │           ├── index.js
│   │   │           ├── s2s.css
│   │   │           ├── s2s.js
│   │   │           ├── components/
│   │   │           │   ├── eventDisplay.css
│   │   │           │   ├── eventDisplay.js
│   │   │           │   ├── meter.js
│   │   │           │   └── settings.js
│   │   │           ├── helper/
│   │   │           │   ├── audioHelper.js
│   │   │           │   ├── audioPlayer.js
│   │   │           │   ├── audioPlayerProcessor.worklet.js
│   │   │           │   ├── config.js
│   │   │           │   └── s2sEvents.js
│   │   │           └── static/
│   │   │               ├── ai_chat_icon.svg
│   │   │               ├── delete.png
│   │   │               ├── demo.jpg
│   │   │               └── kb-customer-avatar.svg
│   │   ├── test/
│   │   │   ├── .env.test
│   │   │   ├── .gitignore
│   │   │   └── e2e/
│   │   │       ├── README.md
│   │   │       ├── generic-test-flow.spec.d.ts
│   │   │       ├── generic-test-flow.spec.ts
│   │   │       └── helpers/
│   │   │           ├── test-data.helper.d.ts
│   │   │           ├── test-data.helper.ts
│   │   │           ├── test.helper.d.ts
│   │   │           └── test.helper.ts
│   │   ├── test-data/
│   │   │   ├── CONFIG_TEMPLATE.md
│   │   │   ├── interview_practice_assistant/
│   │   │   │   ├── 01_TellMeAboutYourself.wav
│   │   │   │   ├── 02_CanYouDescribeAComplexCloudproject.wav
│   │   │   │   ├── 03_CloudMigrationProject.wav
│   │   │   │   ├── 04_HowHaveYouLedTechnicalTeams.wav
│   │   │   │   ├── 05_WhatExcitesYouAboutThisRole.wav
│   │   │   │   ├── README.md
│   │   │   │   └── config.json
│   │   │   ├── order_assistant/
│   │   │   │   ├── 01_ordering_pizza.wav
│   │   │   │   ├── 02_what_specialties.wav
│   │   │   │   ├── 03_bbq_chicken_ranch_interesting.wav
│   │   │   │   ├── 04_get_extra_large.wav
│   │   │   │   ├── 05_thin_and_crispy.wav
│   │   │   │   ├── 06_wings_flavors.wav
│   │   │   │   ├── 07_buffalo_wings.wav
│   │   │   │   ├── 08_delivery.wav
│   │   │   │   ├── PizzabotPromptFlow.docx
│   │   │   │   ├── README.md
│   │   │   │   ├── config.json
│   │   │   │   └── config.json.backup2
│   │   │   └── survey_assistant/
│   │   │       ├── 01_Hello.wav
│   │   │       ├── 02_HappyToHelp.wav
│   │   │       ├── 03_Give4.wav
│   │   │       ├── 04_ReallyNormal.wav
│   │   │       ├── 05_HandlesAccents.wav
│   │   │       ├── 06_PausesOdd.wav
│   │   │       ├── 07_Welcome.wav
│   │   │       ├── README.md
│   │   │       ├── SurveyPromptFlow.docx
│   │   │       └── config.json
│   │   └── test-evaluation-results/
│   │       └── sample/
│   │           ├── complete_results.json
│   │           ├── evaluation_report.md
│   │           ├── evaluation_visualizations.png
│   │           └── run_1_results.json
│   │
│   ├── 04-06-automated-reasoning-evaluations/
│   │   ├── .gitignore
│   │   ├── 04-06-01-automated-reasoning-evaluation.ipynb
│   │   ├── README.md
│   │   ├── housing-code.pdf
│   │   ├── requirements.txt
│   │   └── data/
│   │       ├── ar_tests.json
│   │       ├── ar_tests_essential.json
│   │       └── housing_code_structured_rules.md
│   │
│   ├── 04-07-Multiagent-shared-context-evaluation/
│   │   └── with-strands-agent/
│   │       └── metrics/
│   │           ├── 01-hub-spoke-local-memory.ipynb
│   │           ├── 02-hub-spoke-agentcore-memory.ipynb
│   │           ├── 03-peer-to-peer-dynamic-swarm.ipynb
│   │           ├── 04-peer-to-peer-sequential.ipynb
│   │           ├── 05-conflict-detection-and-visuals.ipynb
│   │           ├── README.md
│   │           ├── eval_helpers.py
│   │           ├── metrics_collector.py
│   │           ├── model_config.py
│   │           ├── requirements.txt
│   │           └── sample-arch-diagram.png
│   │
│   ├── 04-08-Tool-Calling/
│   │   ├── 04-08-Tool-Calling-Evaluation.ipynb
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── data/
│   │       ├── recorded_traces.json
│   │       ├── test_cases.json
│   │       └── tool_definitions.json
│   │
│   ├── 04-11-chatbot/
│   │   ├── .gitignore
│   │   ├── 04-11-01-intro-and-setup.ipynb
│   │   ├── 04-11-02-strands-simulation.ipynb
│   │   ├── 04-11-03-deepeval-simulation.ipynb
│   │   ├── 04-11-04-deepeval-metrics.ipynb
│   │   ├── 04-11-05-strands-evaluators.ipynb
│   │   ├── 04-11-06-synthetic-data.ipynb
│   │   ├── 04-11-07-tool-simulation.ipynb
│   │   ├── 04-11-08-e2e-pipeline.ipynb
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── img/
│   │       └── deepeval_sim.png
│   │
│   └── 04-12-red-teaming/
│       ├── README.md
│       ├── 04-12-01-llm-app-red-teaming/
│       │   ├── 04-12-01-llm-app-red-teaming.ipynb
│       │   └── README.md
│       ├── 04-12-02-testing-bedrock-guardrails/
│       │   ├── 04-12-02-testing-bedrock-guardrails.ipynb
│       │   ├── README.md
│       │   └── requirements.txt
│       ├── 04-12-03-RAG-red-teaming/
│       │   ├── 04-12-03-RAG-red-teaming.ipynb
│       │   ├── README.md
│       │   └── requirements.txt
│       └── 04-12-04-agent-red-teaming/
│           ├── 04-12-04-agent-red-teaming.ipynb
│           ├── README.md
│           └── requirements.txt
│
├── 05-framework-specific-evaluations/
│   ├── .gitkeep
│   ├── 05-01-Prompt-Foo/
│   │   ├── .gitkeep
│   │   ├── 05-01-Promptfoo-basic.ipynb
│   │   └── README.md
│   ├── 05-02-AgentCore/
│   │   ├── .gitkeep
│   │   ├── 05-02-01-Agentic-Metrics-AgentCore.ipynb
│   │   ├── 05-02-02-Agent-and-tool-evals-with-cwlogs.ipynb
│   │   ├── 05-02-03-requirements.txt
│   │   ├── 05-02-04-optional-clean-notebooks.py
│   │   ├── README.md
│   │   └── images/
│   │       ├── Citysearch-AgentCore-Obs-1.png
│   │       ├── Citysearch-AgentCore-Obs-2.png
│   │       ├── Citysearch-AgentCore-Obs-3.png
│   │       └── Citysearch-AgentCore-Obs-4.png
│   ├── 05-03-Strands/
│   │   ├── .gitkeep
│   │   ├── 05-03-Strands-Evals.ipynb
│   │   ├── README.md
│   │   ├── city_pop.csv
│   │   └── city_population_evaluation.json
│   ├── 05-04-AgentCore-Runtime-Evals/
│   │   ├── .gitkeep
│   │   ├── 05-04-01-AgentCore-Evals-Ground-Truth.ipynb
│   │   ├── 05-04-02-AgentCore-Runtime-on-demand-Evals.ipynb
│   │   ├── 05-04-03-AgentCore-Runtime-Online-Evals.ipynb
│   │   ├── README.md
│   │   ├── city_pop.csv
│   │   ├── requirements.txt
│   │   └── images/
│   │       ├── img01.png
│   │       ├── img02.png
│   │       ├── img03.png
│   │       ├── img04.png
│   │       └── img05.png
│   ├── 05-05-DSPy/
│   │   ├── 05-05-DSPy-Prompt-Optimization.ipynb
│   │   ├── README.md
│   │   ├── city_pop.csv
│   │   └── requirements.txt
│   └── 05-06-Mlflow/
│       ├── .env.example
│       ├── .gitignore
│       ├── 05-06-01-Mlflow-Evaluation.ipynb
│       ├── MLFlowApp.png
│       ├── README.md
│       └── requirements.txt
│
├── 06-interactive/
│   ├── README.md
│   ├── 01-operational-metrics/
│   │   └── SKILL.md
│   ├── 02-quality-metrics/
│   │   └── SKILL.md
│   ├── 03-agentic-metrics/
│   │   └── SKILL.md
│   ├── 04-workload-evals/
│   │   ├── SKILL-guardrails.md
│   │   ├── SKILL-rag-evaluation.md
│   │   ├── SKILL-speech-reasoning.md
│   │   └── SKILL-structured-data.md
│   ├── 05-framework-evals/
│   │   ├── SKILL-agentcore.md
│   │   ├── SKILL-dspy.md
│   │   ├── SKILL-promptfoo.md
│   │   └── SKILL-strands.md
│   ├── claude/
│   │   └── kiro.md
│   └── scripts/
│       └── validate_skills.sh
│
└── 07-understanding-failures/
    ├── 01_Discovering_Failure_Patterns.ipynb
    ├── README.md
    └── data/
        └── raw_traces.json
```

---

## Phase 2: Move Log

| # | Old Path | New Path | Notes |
|---|----------|----------|-------|
| 1 | `01-operational-metrics/` | `Foundational Evaluations/01-operational-metrics/` | Entire directory moved |
| 2 | `02-quality-metrics/` | `Foundational Evaluations/02-quality-metrics/` | Entire directory moved |
| 3 | `03-agentic-metrics/` | `Foundational Evaluations/03-agentic-metrics/` | Entire directory moved |
| 4 | `07-understanding-failures/` | `Foundational Evaluations/07-understanding-failures/` | Entire directory moved |
| 5 | `04-workload-specific-evaluations/` | `Workload Specific Evaluations/` | Renamed (removed numbering, spaces, capitalized) |
| 6 | `05-framework-specific-evaluations/` | `Framework Specific Evaluations/` | Renamed (removed numbering, spaces, capitalized) |
| 7 | `06-interactive/` | `Interactive Learning/` | Renamed (removed numbering, spaces, capitalized) |
| 8 | `Foundational Evaluations/07-understanding-failures/` | `Foundational Evaluations/04-understanding-failures/` | Renumbered from 07 to 04 |
| 9 | `Foundational Evaluations/03-agentic-metrics/` | `Foundational Evaluations/04-agentic-metrics/` | Swapped: renumbered 03 → 04 |
| 10 | `Foundational Evaluations/04-understanding-failures/` | `Foundational Evaluations/03-understanding-failures/` | Swapped: renumbered 04 → 03 |
| 11 | `Foundational Evaluations/04-agentic-metrics/03-Agentic-Metrics.ipynb` | `Foundational Evaluations/04-agentic-metrics/01-Agentic-Metrics.ipynb` | Renumbered notebook to match new directory scheme |
| 12 | `Framework Specific Evaluations/05-01-Prompt-Foo/` | `Framework Specific Evaluations/Prompt Foo/` | Removed numbering, spaces instead of dashes |
| 13 | `Framework Specific Evaluations/05-02-AgentCore/` | `Framework Specific Evaluations/AgentCore/` | Removed numbering |
| 14 | `Framework Specific Evaluations/05-03-Strands/` | `Framework Specific Evaluations/Strands/` | Removed numbering |
| 15 | `Framework Specific Evaluations/05-04-AgentCore-Runtime-Evals/` | `Framework Specific Evaluations/AgentCore Runtime Evals/` | Removed numbering, spaces instead of dashes |
| 16 | `Framework Specific Evaluations/05-05-DSPy/` | `Framework Specific Evaluations/DSPy/` | Removed numbering |
| 17 | `Framework Specific Evaluations/05-06-Mlflow/` | `Framework Specific Evaluations/Mlflow/` | Removed numbering |
| 18 | `Framework Specific Evaluations/Prompt Foo/05-01-Promptfoo-basic.ipynb` | `Framework Specific Evaluations/Prompt Foo/01 Promptfoo basic.ipynb` | Stripped 05-01- prefix, dashes to spaces |
| 19 | `Framework Specific Evaluations/AgentCore/05-02-01-Agentic-Metrics-AgentCore.ipynb` | `Framework Specific Evaluations/AgentCore/01 Agentic Metrics AgentCore.ipynb` | Stripped 05-02- prefix, dashes to spaces |
| 20 | `Framework Specific Evaluations/AgentCore/05-02-02-Agent-and-tool-evals-with-cwlogs.ipynb` | `Framework Specific Evaluations/AgentCore/02 Agent and tool evals with cwlogs.ipynb` | Stripped 05-02- prefix, dashes to spaces |
| 21 | `Framework Specific Evaluations/AgentCore/05-02-03-requirements.txt` | `Framework Specific Evaluations/AgentCore/03 requirements.txt` | Stripped 05-02- prefix, dashes to spaces |
| 22 | `Framework Specific Evaluations/AgentCore/05-02-04-optional-clean-notebooks.py` | `Framework Specific Evaluations/AgentCore/04 optional clean notebooks.py` | Stripped 05-02- prefix, dashes to spaces |
| 23 | `Framework Specific Evaluations/Strands/05-03-Strands-Evals.ipynb` | `Framework Specific Evaluations/Strands/01 Strands Evals.ipynb` | Stripped 05-03- prefix, dashes to spaces |
| 24 | `Framework Specific Evaluations/AgentCore Runtime Evals/05-04-01-AgentCore-Evals-Ground-Truth.ipynb` | `Framework Specific Evaluations/AgentCore Runtime Evals/01 AgentCore Evals Ground Truth.ipynb` | Stripped 05-04- prefix, dashes to spaces |
| 25 | `Framework Specific Evaluations/AgentCore Runtime Evals/05-04-02-AgentCore-Runtime-on-demand-Evals.ipynb` | `Framework Specific Evaluations/AgentCore Runtime Evals/02 AgentCore Runtime on demand Evals.ipynb` | Stripped 05-04- prefix, dashes to spaces |
| 26 | `Framework Specific Evaluations/AgentCore Runtime Evals/05-04-03-AgentCore-Runtime-Online-Evals.ipynb` | `Framework Specific Evaluations/AgentCore Runtime Evals/03 AgentCore Runtime Online Evals.ipynb` | Stripped 05-04- prefix, dashes to spaces |
| 27 | `Framework Specific Evaluations/DSPy/05-05-DSPy-Prompt-Optimization.ipynb` | `Framework Specific Evaluations/DSPy/01 DSPy Prompt Optimization.ipynb` | Stripped 05-05- prefix, dashes to spaces |
| 28 | `Framework Specific Evaluations/Mlflow/05-06-01-Mlflow-Evaluation.ipynb` | `Framework Specific Evaluations/Mlflow/01 Mlflow Evaluation.ipynb` | Stripped 05-06- prefix, dashes to spaces |
| 29 | `Workload Specific Evaluations/04-01-Intelligent-Document-Processing/` | `Workload Specific Evaluations/Intelligent Document Processing/` | Removed numbering, dashes to spaces |
| 30 | `Workload Specific Evaluations/04-02-Guardrails/` | `Workload Specific Evaluations/Guardrails/` | Removed numbering |
| 31 | `Workload Specific Evaluations/04-03-Basic-RAG/` | `Workload Specific Evaluations/Basic RAG/` | Removed numbering, dashes to spaces |
| 32 | `Workload Specific Evaluations/04-04-MultiModal-RAG/` | `Workload Specific Evaluations/MultiModal RAG/` | Removed numbering, dashes to spaces |
| 33 | `Workload Specific Evaluations/04-05-Speech-to-Speech/` | `Workload Specific Evaluations/Speech to Speech/` | Removed numbering, dashes to spaces |
| 34 | `Workload Specific Evaluations/04-06-automated-reasoning-evaluations/` | `Workload Specific Evaluations/Automated Reasoning Evaluations/` | Removed numbering, dashes to spaces, capitalized |
| 35 | `Workload Specific Evaluations/04-07-Multiagent-shared-context-evaluation/` | `Workload Specific Evaluations/Multiagent Shared Context Evaluation/` | Removed numbering, dashes to spaces, capitalized |
| 36 | `Workload Specific Evaluations/04-08-Tool-Calling/` | `Workload Specific Evaluations/Tool Calling/` | Removed numbering, dashes to spaces |
| 37 | `Workload Specific Evaluations/04-11-chatbot/` | `Workload Specific Evaluations/Chatbot/` | Removed numbering, capitalized |
| 38 | `Workload Specific Evaluations/04-12-red-teaming/` | `Workload Specific Evaluations/Red Teaming/` | Removed numbering, dashes to spaces, capitalized |
| 39 | `Workload Specific Evaluations/Intelligent Document Processing/04-01-Simple-structured-data-evaluation.ipynb` | `Workload Specific Evaluations/Intelligent Document Processing/01 Simple structured data evaluation.ipynb` | Stripped 04-01- prefix, dashes to spaces |
| 40 | `Workload Specific Evaluations/Guardrails/04-02-01-filters.ipynb` | `Workload Specific Evaluations/Guardrails/01 filters.ipynb` | Stripped 04-02- prefix, dashes to spaces |
| 41 | `Workload Specific Evaluations/Guardrails/04-02-02-grounding.ipynb` | `Workload Specific Evaluations/Guardrails/02 grounding.ipynb` | Stripped 04-02- prefix, dashes to spaces |
| 42 | `Workload Specific Evaluations/Guardrails/04-02-03-alignment.ipynb` | `Workload Specific Evaluations/Guardrails/03 alignment.ipynb` | Stripped 04-02- prefix, dashes to spaces |
| 43 | `Workload Specific Evaluations/Guardrails/04-02-04-operational.ipynb` | `Workload Specific Evaluations/Guardrails/04 operational.ipynb` | Stripped 04-02- prefix, dashes to spaces |
| 44 | `Workload Specific Evaluations/Guardrails/04-02-05-reasoning.md` | `Workload Specific Evaluations/Guardrails/05 reasoning.md` | Stripped 04-02- prefix, dashes to spaces |
| 45 | `Workload Specific Evaluations/Guardrails/04-02-06-evaluation.ipynb` | `Workload Specific Evaluations/Guardrails/06 evaluation.ipynb` | Stripped 04-02- prefix, dashes to spaces |
| 46 | `Workload Specific Evaluations/Basic RAG/example-notebook/04-03-Basic-RAG-Evaluation.ipynb` | `Workload Specific Evaluations/Basic RAG/example-notebook/01 Basic RAG Evaluation.ipynb` | Stripped 04-03- prefix, dashes to spaces |
| 47 | `Workload Specific Evaluations/MultiModal RAG/04-04-01-Multimodal-RAG.ipynb` | `Workload Specific Evaluations/MultiModal RAG/01 Multimodal RAG.ipynb` | Stripped 04-04- prefix, dashes to spaces |
| 48 | `Workload Specific Evaluations/Automated Reasoning Evaluations/04-06-01-automated-reasoning-evaluation.ipynb` | `Workload Specific Evaluations/Automated Reasoning Evaluations/01 automated reasoning evaluation.ipynb` | Stripped 04-06- prefix, dashes to spaces |
| 49 | `Workload Specific Evaluations/Tool Calling/04-08-Tool-Calling-Evaluation.ipynb` | `Workload Specific Evaluations/Tool Calling/01 Tool Calling Evaluation.ipynb` | Stripped 04-08- prefix, dashes to spaces |
| 50 | `Workload Specific Evaluations/Chatbot/04-11-01-intro-and-setup.ipynb` | `Workload Specific Evaluations/Chatbot/01 intro and setup.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 51 | `Workload Specific Evaluations/Chatbot/04-11-02-strands-simulation.ipynb` | `Workload Specific Evaluations/Chatbot/02 strands simulation.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 52 | `Workload Specific Evaluations/Chatbot/04-11-03-deepeval-simulation.ipynb` | `Workload Specific Evaluations/Chatbot/03 deepeval simulation.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 53 | `Workload Specific Evaluations/Chatbot/04-11-04-deepeval-metrics.ipynb` | `Workload Specific Evaluations/Chatbot/04 deepeval metrics.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 54 | `Workload Specific Evaluations/Chatbot/04-11-05-strands-evaluators.ipynb` | `Workload Specific Evaluations/Chatbot/05 strands evaluators.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 55 | `Workload Specific Evaluations/Chatbot/04-11-06-synthetic-data.ipynb` | `Workload Specific Evaluations/Chatbot/06 synthetic data.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 56 | `Workload Specific Evaluations/Chatbot/04-11-07-tool-simulation.ipynb` | `Workload Specific Evaluations/Chatbot/07 tool simulation.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 57 | `Workload Specific Evaluations/Chatbot/04-11-08-e2e-pipeline.ipynb` | `Workload Specific Evaluations/Chatbot/08 e2e pipeline.ipynb` | Stripped 04-11- prefix, dashes to spaces |
| 58 | `Workload Specific Evaluations/Red Teaming/04-12-01-llm-app-red-teaming/` | `Workload Specific Evaluations/Red Teaming/01 LLM App Red Teaming/` | Stripped 04-12- prefix, dashes to spaces, capitalized |
| 59 | `Workload Specific Evaluations/Red Teaming/04-12-02-testing-bedrock-guardrails/` | `Workload Specific Evaluations/Red Teaming/02 Testing Bedrock Guardrails/` | Stripped 04-12- prefix, dashes to spaces, capitalized |
| 60 | `Workload Specific Evaluations/Red Teaming/04-12-03-RAG-red-teaming/` | `Workload Specific Evaluations/Red Teaming/03 RAG Red Teaming/` | Stripped 04-12- prefix, dashes to spaces, capitalized |
| 61 | `Workload Specific Evaluations/Red Teaming/04-12-04-agent-red-teaming/` | `Workload Specific Evaluations/Red Teaming/04 Agent Red Teaming/` | Stripped 04-12- prefix, dashes to spaces, capitalized |
| 62 | `Workload Specific Evaluations/Red Teaming/01 LLM App Red Teaming/04-12-01-llm-app-red-teaming.ipynb` | `Workload Specific Evaluations/Red Teaming/01 LLM App Red Teaming/01 llm app red teaming.ipynb` | Stripped 04-12- prefix, dashes to spaces |
| 63 | `Workload Specific Evaluations/Red Teaming/02 Testing Bedrock Guardrails/04-12-02-testing-bedrock-guardrails.ipynb` | `Workload Specific Evaluations/Red Teaming/02 Testing Bedrock Guardrails/01 testing bedrock guardrails.ipynb` | Stripped 04-12- prefix, dashes to spaces |
| 64 | `Workload Specific Evaluations/Red Teaming/03 RAG Red Teaming/04-12-03-RAG-red-teaming.ipynb` | `Workload Specific Evaluations/Red Teaming/03 RAG Red Teaming/01 RAG red teaming.ipynb` | Stripped 04-12- prefix, dashes to spaces |
| 65 | `Workload Specific Evaluations/Red Teaming/04 Agent Red Teaming/04-12-04-agent-red-teaming.ipynb` | `Workload Specific Evaluations/Red Teaming/04 Agent Red Teaming/01 agent red teaming.ipynb` | Stripped 04-12- prefix, dashes to spaces |
| 66 | `Workload Specific Evaluations/Speech to Speech/s2s_entire_eval_pipeline.ipynb` | `Workload Specific Evaluations/Speech to Speech/01 s2s_entire_eval_pipeline.ipynb` | Added 01 prefix for consistency |
| 67 | `Interactive Learning/01-operational-metrics/SKILL.md` | `Interactive Learning/foundational evaluations/SKILL-operational.md` | Moved to new subdir, renamed |
| 68 | `Interactive Learning/02-quality-metrics/SKILL.md` | `Interactive Learning/foundational evaluations/SKILL-quality.md` | Moved to new subdir, renamed |
| 69 | `Interactive Learning/03-agentic-metrics/SKILL.md` | `Interactive Learning/foundational evaluations/SKILL-agentic.md` | Moved to new subdir, renamed |
| 70 | `Interactive Learning/01-operational-metrics/` | *(deleted)* | Empty after move |
| 71 | `Interactive Learning/02-quality-metrics/` | *(deleted)* | Empty after move |
| 72 | `Interactive Learning/03-agentic-metrics/` | *(deleted)* | Empty after move |
| 73 | `Interactive Learning/04-workload-evals/` | `Interactive Learning/workload evals/` | Removed numbering, dashes to spaces |
| 74 | `Interactive Learning/05-framework-evals/` | `Interactive Learning/framework evals/` | Removed numbering, dashes to spaces |

---

## Phase 3: Link Update Checklist

After all moves are complete, each file containing inter-repo links will be reviewed and updated.

| File | Link Found | Old Target | New Target | Updated? |
|------|-----------|------------|------------|----------|
| | | | | |

*(Will be populated during the link-fixing phase.)*
