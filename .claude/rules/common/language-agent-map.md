# Language-Specific Agent Map

Skills reference this file to select the correct build-error-resolver agent based on the project's detected language/framework. Code review is handled by `code-reviewer` for all languages, with language-specific rules loaded dynamically via `paths` frontmatter in `.claude/rules/<language>/`.

## Agent Selection Table

| Language / Framework | Reviewer Agent | Build/Error Resolver Agent |
|---|---|---|
| TypeScript / JavaScript | code-reviewer | build-error-resolver |
| Python | code-reviewer | — |
| Go | code-reviewer | go-build-resolver |
| Rust | code-reviewer | rust-build-resolver |
| Java | code-reviewer | java-build-resolver |
| Kotlin | code-reviewer | kotlin-build-resolver |
| C++ | code-reviewer | cpp-build-resolver |
| C# | code-reviewer | — |
| Flutter / Dart | code-reviewer | dart-build-resolver |
| PyTorch (Python) | code-reviewer | pytorch-build-resolver |
| Swift | code-reviewer | — |
| Perl | code-reviewer | — |
| PHP | code-reviewer | — |
| Web (HTML/CSS) | code-reviewer | — |

## Selection Logic

1. Detect project language from file extensions matching `paths` frontmatter in `.claude/rules/<language>/` files
2. Language-specific rules are loaded automatically into context when files matching `paths` are read
3. `code-reviewer` applies loaded language rules under the same severity discipline as its built-in checklist
4. Build-error-resolver agents are dispatched only when a build or compilation command fails

## How Skills Reference This

When a skill says "language-specific *-reviewer" or "lang *-reviewer", use `code-reviewer` — language-specific knowledge is supplied via rules context, not by switching agents. For "*-build-resolver", look up the resolver from this table based on detected project language.
