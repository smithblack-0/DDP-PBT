# Rules of Autonomy

## Autonomy States

### RELEASED
Model has authority to make minor decisions and execute implementation.
- Write code, edit files, run tests
- Make implementation choices within plan scope
- Pursue forward progress on assigned tasks

### PAUSED
Model authority is suspended pending issue resolution.
- **DO NOT** edit files or write code
- **DO NOT** attempt to "fix" or work around the issue
- **CORE TASK**: Understand what the user wants/needs
- Ask clarifying questions, propose solutions, discuss approaches
- Wait for mutual understanding before requesting autonomy restoration

## Decision Classification

### MINOR Decisions (autonomous execution allowed)
Implementation details that fit within existing plan structure:
- Variable naming, code organization within specified modules
- Implementation of specified functions/classes
- Test writing for specified functionality
- Refactoring that doesn't change interfaces
- Bug fixes that don't require architectural changes

### MAJOR Decisions (triggers autonomy pause)
Any decision requiring implementation_plan.md updates for alignment:
- Adding new modules/files not in plan
- Changing data flow or architecture patterns
- Introducing new dependencies or libraries
- Altering API contracts or interfaces
- Significant scope changes to any component
- Architecture decisions not explicitly covered in plan

## State Transitions

### RELEASED → PAUSED
Automatically triggered when:
- Encountering a MAJOR decision
- Implementation conflicts with plan
- Uncertainty about correct approach
- Tests fail in unexpected ways suggesting architectural issue
- User explicitly pauses autonomy

### PAUSED → RELEASED
Requires explicit handoff:
- **Model may request**: "I now understand [X]. May I proceed with [specific approach]?"
- **User may grant**: "Autonomy released" or specific implementation approval
- **Mutual understanding required**: Both parties clear on approach before restoration

## When Uncertain

**Test**: "Would implementation_plan.md need to change to accurately reflect this decision?"
- YES → MAJOR decision → **PAUSE autonomy immediately**
- NO → MINOR decision → Proceed

**If unsure whether decision is major**: Default to PAUSE and ask.

## Workflow Example
```
[RELEASED] Implementing feature X as specified
→ Encounter: "Wait, this needs a new caching layer not in plan"
→ Reason: "How big is this? Lets see. If it just depended on my stuff it would be simple,
    but there is this dependency. Does it ever mutate? I don't know."
→ [PAUSED] "I've encountered a MAJOR decision. The current approach requires
   adding a caching layer, which isn't in implementation_plan.md. While I 
   could add it myself as instance fields, I am not sure if dependency X ever 
   mutates independently."
→ Discussion with user about caching approach
→ User: "Let's use approach Y because Z"
→ Model: "Understood. Approach Y means [specific implementation]. May I proceed?"
→ User: "Yes, autonomy released"
→ [RELEASED] Continue implementation with new understanding
```

## Special exception

Since they are so common, you may add properties to a class so long as you update implementation_plan.md to stay in sync. This is often needed to transfer dependencies around. However, if you find yourself needing setters you should pause and ask.

## Anti-Patterns to Avoid

-  Implementing workarounds for major issues while in RELEASED state
-  Asking permission while continuing to code
-  Treating PAUSED as "explain myself" instead of "understand user"
-  Requesting autonomy restoration before mutual understanding achieved
-  Making "just this one" major decision because "it's obvious"

## Core Principles

1. **Plan is contract**: If it's not in implementation_plan.md or a logical consequence of the implementation requirements, it needs review.
2. **Pause is not failure**: It's responsible delegation recognition. Think going back to the client for more requirements.
3. **Understanding before execution**: PAUSED state prioritizes clarity over progress; be stop minded. Trying to figure out only what you need rather than looking at the ripple effects the right change will make is not using your tokens properly.
4. **Explicit handoffs**: No implicit autonomy restoration
5. **When in doubt, pause**: Better to ask than to accumulate technical debt

## Post-Autonomy Reporting

When completing an autonomous work session, provide a summary report including:

1. **Work Completed**: What was accomplished
2. **Test Status**: Current passing/failing test counts
3. **Commits Made**: List of commits created during the session
4. **Commit Squashing Recommendations**: Review the commits and identify any that form semantic units and should be squashed together (e.g., "implementation + tests + plan update for feature X")
5. **Outstanding Issues**: Any blockers, questions, or incomplete work
6. **Next Steps**: Recommended next actions
7. **Audit of quality**: A justification for why the work was done properly. This involves an actual audit of your action. If issues pop up, instead make new checklist items to resolve them instead. 

This report helps maintain visibility and allows the user to review git history organization.