# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-01-21

Code formatting and style cleanup.

- Run ruff and black across entire codebase
- Fix remaining formatting issues (unused variables, line length)

## [0.3.1] - 2026-01-21

Post-release fixes and documentation.

- Fix AbstractStrategy tests with distributed context fixture
- Fix TopPopulationSexualStrategy factory to return correct type
- Fix Crossbreeder tests for new API
- Update CLAUDE.md autonomy rules and context file reading guidance
- Add white box testing design issue clarification

## [0.3.0] - 2026-01-20

All strategy implementations complete with comprehensive test coverage.

- TopKStrategy comprehensive test suite
- WeightedAverageStrategy comprehensive test suite
- TopPopulationAsexualStrategy comprehensive test suite
- TopPopulationSexualStrategy comprehensive test suite
- Fix np.random.choice usage in TopK and PopulationAsexual strategies
- Add critical reminder to read context files entirely (CLAUDE.md)

## [0.2.7] - 2026-01-20

Integration testing and strategy refinements.

- Add integration test for TopScoreStrategy with AdamW
- Fix test mocking to avoid distributed requirements
- Fix TopScoreStrategy schema setup and documentation
- Refactor crossbreeding implementation
- Add .gitignore for Python project

## [0.2.6] - 2026-01-20

Test improvements and TopScoreStrategy fixes.

- Fix TopScoreStrategy score() signature to use validation_metrics list
- Fix TopScoreStrategy to require perturber and always use it
- Fix TopScoreStrategy factory parameters
- Add comprehensive tests for Communication properties
- Add test for Communication single worker rejection
- Fix test configurations

## [0.2.5] - 2026-01-20

Systematic Permuter to Perturber rename.

- Rename across implementation_plan.md, crossbreeder, strategies, and all test files
- Update docstrings to reflect new naming

## [0.2.4] - 2026-01-20

Communication enhancements and fixes.

- Add rank and world_size properties to Communication
- Fix Crossbreeder API signature to accept validation_metrics
- Template improvements for concrete strategy implementations
- Various minor fixes and updates

## [0.2.3] - 2026-01-20

Contract clarification and API improvements.

- Clarify strategy contract in implementation_plan.md
- Fix validation_metrics API to use list instead of individual metrics
- TopScoreStrategy contract compliance fixes
- Fix typing oversights
- Add comment on major testing oversight
- Various minor tweaks

## [0.2.2] - 2026-01-20

Import pattern standardization and file reorganization.

- Standardize import patterns to absolute imports with src. prefix
- Reorganize test file structure
- Move files into correct directories

## [0.2.1] - 2026-01-20

Crossbreeder refactor and project reorganization.

- Large Crossbreeder rebuild for clarity and simplification
- Move strategy implementations into strategies folder
- TopScoreStrategy contract compliance fixes

## [0.2.0] - 2026-01-20

TopScoreStrategy complete.

- Comprehensive test suite for TopScoreStrategy
- TopScoreStrategy implementation with factory pattern
- Fix floating point precision in perturber tests
- Fix test_setup_schema_stores_reference
- Remove incorrect WeightedAverage description from implementation_plan.md

## [0.1.2] - 2026-01-20

Documentation and test improvements.

- Add post-autonomy reporting requirements to CLAUDE.md
- Add import style guide to implementation_plan.md
- Fix perturber tests: add required min field for log-type schemas
- Additional CLAUDE.md tweaks

## [0.1.1] - 2026-01-20

Post-setup refinements and fixes.

- Perturber component refinement (improved responsibility separation and sanity checking)
- Early bugfixes across base components
- Minor tweaks from initial audit
- Add autonomy rules to CLAUDE.md

## [0.1.0] - 2026-01-19

Initial project setup.

- CI/CD workflows (GitHub Actions for testing, release validation)
- Project configuration (pyproject.toml with dependencies, black/ruff config)
- Base component implementations (State, Perturber, Crossbreeder, Communication, AbstractStrategy)
- Comprehensive test suites for all base components
- Documentation (README, implementation_plan.md) 
