# Core Rules
0. **Bootstrap Requirement:** This `GEMINI.md` file must be read and its rules acknowledged as a bootstrap process whenever a new session or sub-agent is initialized.
1. **Knowledge Persistence:** 
    - **Common Knowledge:** `GEMINI/Architecture/` represents the current system-wide state of the codebase. It MUST only be updated after an issue is officially concluded (final verdict reached) and a clear directive to "update the knowledge" is issued. **NEVER update architecture docs during an active research or implementation phase.**
    - **Active Development:** ALL architectural design changes, implementations, and findings during an active task MUST remain exclusively within the task's folder (e.g., `GEMINI/Issues/<issue-id>/`). This ensures the common knowledge folder remains a stable, immutable reference until a change is verified and finalized.
2. **Issue-Based Tracking:** Every significant task or bug must have a dedicated folder in `GEMINI/Issues/` (e.g., `GEMINI/Issues/213-canvas-virtualization`).
    - ALL lifecycle documents (Analysis, Design, Planning, Implementation) for the issue MUST stay in this folder.
    - NEVER update global architecture files based on uncommitted or in-progress implementations.
3. **Phase-Gate Lifecycle:** Each issue must progress through documented phases:
   - **Analysis:** Requirements gathering, root cause analysis, current state mapping.
   - **Design:** Architectural proposals, trade-offs, and UI/UX impact.
   - **Planning:** Step-by-step execution plan with validation gates.
   - **Implementation:** Progress reports and final verification details.
- **Continuous Integration:** 
    - **Frequent Compilation:** Perform a compilation check after every significant code change to catch syntax and dependency errors early.
    - **Validation:** Verification is mandatory for task closure. For bug fixes, empirically reproduce the failure with a test case before applying the fix. Due to the nature of the graphical editor, compilation is NOT enough for finality; the user MUST manually verify each implementation attempt before a final verdict is reached.

# Conventions & Style
- **Authorship:** 
    - NEVER modify the `@author` tag in existing files.
    - ALWAYS use `@author Hoai Thu Nguyen` for any newly created files.
- **Internationalization (i18n):** When refactoring a class, identify strings not intended for user display (e.g., logging, file paths, regex). Mark these as non-translatable by appending the line comment `//$NON-NLS-1$`. This is part of a gradual iteration to account for strings that do not require translation.
- **javadoc and comment style:** briefly add javadoc to created class. Refact existed javadoc if core design changed. Javadoc on method is a MUST (brief description and params and return if needed). Comments on code are encouraged but only on important code, not something trivial like "// Prepare attributes". Also discourage to remove existed comments. Don't use numbering comment in a logic chain. Don't use "you" in comments.
- **Import datatype:** avoid using full path datatype, import it at the head instead.
- **Surgical Updates:** Always prefer `replace` for targeted edits to documentation as new knowledge is acquired.

# Directory Structure
- `GEMINI/Architecture/`: Plugin-specific and system-wide documentation.
- `GEMINI/Issues/`: Task-specific lifecycle documentation.
- `GEMINI/Tmp/`: Temporary scratchpads or session-specific data.

# Behavioral guidelines
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
