# AFD Plugin Development Guidelines

This document provides instructions for contributors to the AFD Plugin project.
Follow these guidelines to keep code changes maintainable, focused, and
consistent.

## Code Style

### Python Conventions

- **Imports**: All imports at the top of the file. Valid exceptions for
  inline/deferred imports:
    - Circular imports (use inline imports)
    - Lazy loading for worker/isolation processes
    - Note: Type-checking imports wrapped in `if TYPE_CHECKING:` should still
      be placed at the top of the file.

- **Global Variables**: Avoid new global variables. Pass dependencies explicitly
  through function parameters.

    **Allowed:**
    - Constants named `ALL_UPPER_CASE`
    - Immutable configuration objects

    **Requires Approval:**
    - Any new mutable global state

- **No Magic Numbers**: Use named constants with descriptive names:

    ```python
    # Bad
    if seq_len > 2048: ...

    # Good
    MAX_CONTEXT_LENGTH = 2048
    if seq_len > MAX_CONTEXT_LENGTH: ...
    ```

- **Descriptive Naming**: Use names that describe functionality, not
  implementation details.

    ```python
    # Bad
    is_deepseek_v3_r1
    flag1
    tmp_var

    # Good
    supports_dynamic_temperature
    uses_speculative_decoding
    ```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `AFDAttentionModelRunner`,
  `AFDNPUAttentionModelRunner`, `GPUFFNModelRunner`)
- **Functions/Methods**: `snake_case` (e.g., `forward_pass`,
  `compute_attention`)
- **Constants**: `ALL_UPPER_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Variables**: `snake_case` (e.g., `token_ids`, `sequence_lengths`)

## Development Notes

- When using data structures from vLLM or vLLM Ascend, access their functions
  and member variables directly. Avoid `getattr`, `hasattr`, or proactively
  raising custom exceptions unless necessary. This keeps upstream compatibility
  issues visible through the original error when upgrading, and allows static
  type checkers such as mypy or pyright to detect missing or renamed attributes
  early.

- Avoid `Any` and `object` in parameter types unless necessary. Prefer concrete
  types that describe the expected contract.

- Do not split simple functions into excessive helper functions. Extract helpers
  only when they reduce meaningful complexity, avoid real duplication, or match
  an established local pattern.

## Patching Requirement

**Strict Review Required**: All new patches must undergo thorough architectural
review.

Reviewers must verify:

- The patch targets the correct upstream component
- The patch is minimal and focused
- Performance implications are understood
- A long-term plan exists for upstream contribution or removal

When copying or wrapping upstream code, mark every AFD-specific difference with
`# ### PATCH START: ...` and `# ### PATCH END: ...` comments. Keep the marker
text short and specific so reviewers can compare the patch against upstream
quickly.

Every patch function must have comments immediately above the function that
explain why that upstream function is patched and what behavior the patch adds
or changes. The patched function signature, including return type, must match
the upstream function exactly. If a patch must add parameters, document those
new parameters in the comments immediately above the patch function.

**Required Pattern**: AFD-specific functionality should be implemented via:

1. **Patching**:
    - `afd_plugin/compat/patches/` - vLLM compatibility patches
    - `afd_plugin/compat/patches/npu/` - NPU-specific compatibility patches
    - `afd_plugin/compat/ascend/` - Ascend runtime compatibility patches
    - Patch is not the best solution for all cases. Use it when necessary.

2. **Inheritance**:
    - Extend vLLM model runners or workers for AFD-specific behavior
    - Add AFD-specific components via composition
    - Keep backend-specific operators in clearly scoped backend modules

3. **External upstream contributions** where appropriate

**Example Patch Pattern:**

```python
# afd_plugin/compat/patches/example_patch.py
from vllm.some_upstream_module import UpstreamClass

_original_method = UpstreamClass.method

# Patch reason: upstream does not know about AFD's routing policy.
# Patch functionality: add AFD routing while delegating non-AFD requests.
# Signature: matches upstream; no added parameters.
def method(self, *args, **kwargs):
    # ### PATCH START: AFD custom routing
    # AFD-specific behavior that differs from upstream.
    ...
    # ### PATCH END: AFD custom routing
    # If delegation is needed:
    # return _original_method(self, *args, **kwargs)

UpstreamClass.method = method  # Patch upstream class
```
