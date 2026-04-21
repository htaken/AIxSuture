# Agent Orchestration

## Skill-First Principle

Skills always lead. Agents are dispatched by skills in a defined order. Agents never take autonomous control of a workflow.

Each skill's SKILL.md defines its own Agent Integration section specifying which agents to call, in what order, and under what conditions. This file provides the master reference.

## Operational Rules

1. **Skills lead, agents follow.** Never dispatch an agent outside the context of a skill's defined chain.
2. **Rigid skills (brainstorming, systematic-debugging, test-driven-development) never yield control to agents.** No agents permitted.
3. **loop-operator is excluded from normal flows.**
4. **requesting-code-review and verification-before-completion benefit most from heavy reviewer stacking.** Stack reviewers generously in these skills.

## Skill → Agent Mapping

| Skill | Agent Chain | Notes |
|-------|-------------|-------|
| using-superpowers | — | Entry skill |
| brainstorming | — | Rigid |
| using-git-worktrees | — | Procedure skill |
| writing-plans | architect → docs-lookup | Design consistency and external spec verification |
| subagent-driven-development | implementer → (lang-*-build-resolver) → spec-reviewer → code-quality-reviewer (code-reviewer internally) | Build-resolver dispatched on build failure before spec review; two-stage review per task; code-quality-reviewer uses code-reviewer template |
| executing-plans | — | No agents. Fallback for no-subagent environments |
| test-driven-development | — | Rigid |
| systematic-debugging | — | Rigid |
| requesting-code-review | security-reviewer → (database-reviewer) → (performance-optimizer) → code-reviewer | High-risk-first order |
| receiving-code-review | — | No agents. Claude evaluates feedback independently |
| verification-before-completion | — | No agents. Verification procedure, Claude-driven |
| finishing-a-development-branch | — | No agents. Verification procedure, Claude-driven |
| dispatching-parallel-agents | architect + *-build-resolver (dynamic) | Per-task selection |

Parentheses `()` indicate optional agents dispatched only under specific conditions.

## Available Agents

### General Purpose

| Agent | Purpose |
|-------|---------|
| architect | System design and architectural decisions |
| code-reviewer | Code review and plan alignment |
| docs-lookup | Library/API documentation lookup |
| doc-updater | Documentation updates |
| e2e-runner | E2E testing with Playwright |
| refactor-cleaner | Dead code cleanup and consolidation |

### Language-Specific

Language-specific code review is handled by `code-reviewer`, which reads language rules from `.claude/rules/<language>/` automatically via `paths` frontmatter when files of that type are under review. See [language-agent-map.md](language-agent-map.md) for build-resolver mapping per language.

### Domain Specialists

| Agent | Purpose |
|-------|---------|
| security-reviewer | Security vulnerability detection (OWASP Top 10) |
| performance-optimizer | Performance analysis and optimization |
| database-reviewer | PostgreSQL schema, query optimization, RLS |

### Parallel Dispatch

For dispatching multiple independent agents concurrently, use the **dispatching-parallel-agents** skill. It handles per-task agent selection using the language-agent-map.
