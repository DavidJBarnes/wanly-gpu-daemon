# wanly-gpu-daemon -- Codebase Scorecard

**Audit Date:** 2026-03-12T15:59:35Z
**Branch:** feature/daemon-safeguards
**Commit:** 3768d97f9d2d644f1a1d0e9aa670f1bd448d47c6
**Auditor:** Claude Code (Automated)

---

## Summary

| Metric | Value | Grade |
|--------|-------|-------|
| Total Source Lines | 2,235 | -- |
| Source Files | 14 (.py) | -- |
| Test Files | 0 | F |
| Test Coverage | 0% (no tests) | F |
| Dependency Pinning | 0/6 pinned | F |
| Type Annotation Coverage | 72% (46/64 functions) | C |
| Docstring Coverage | 72% (46/64 functions) | C |
| TODO/FIXME Count | 0 in code | A |
| print() Statements | 0 | A |
| Logging | Consistent (stdlib logging) | A |
| Error Handling | Comprehensive try/except + retry | A |
| Dockerfile | None | N/A |
| CI/CD | None (deployed via git pull) | D |
| Packaging (pyproject.toml) | Missing | F |

---

## Category Scores

### Code Quality: B

**Strengths:**
- Clean module separation (each file has a clear single responsibility)
- Consistent use of stdlib `logging` (no `print()` calls)
- Proper async patterns throughout (asyncio.Event, signal handlers, graceful shutdown)
- Good error handling with retry logic for transient failures
- Atomic file writes for LoRA/resource downloads
- Progress reporting to API during execution
- VRAM monitoring after each segment for leak detection

**Weaknesses:**
- 28% of functions missing return type annotations
- 28% of functions missing docstrings
- Hardcoded magic numbers for ComfyUI node IDs
- Main loop functions (`heartbeat_loop`, `job_poll_loop`) have many parameters without types

### Architecture: A-

**Strengths:**
- Clean separation: daemon pulls work, never receives pushes
- S3 proxy pattern eliminates AWS credential management on workers
- Drain support for graceful RunPod pod shutdown
- Stale process cleanup on startup
- Pre-flight validation chain (nodes -> resources -> models) with fail-fast behavior
- WebSocket-based execution monitoring (not polling)
- PainterLongVideo identity anchoring for multi-segment consistency

**Weaknesses:**
- `friendly_name_ref` mutable list hack for cross-task name sharing (could use a small state object)
- No retry on ComfyUI submission or execution (only on image downloads)

### Testing: F

No test files exist. Zero coverage.

### Dependency Management: F

- No version pins in requirements.txt
- No pyproject.toml or setup.py
- No lock file
- 6 dependencies all unpinned

### Operational Readiness: B-

**Strengths:**
- Graceful shutdown with 10-minute wait for in-progress work
- Signal handling (SIGINT, SIGTERM)
- Heartbeat with GPU stats reporting
- ComfyUI queue cleanup on startup
- Partial download cleanup
- Model validation before accepting work
- Stale daemon process killer

**Weaknesses:**
- No health check endpoint (daemon is not an HTTP server)
- No structured logging (plain text format)
- No metrics/observability beyond log output
- No Dockerfile (relies on host Python environment)

### Security: B

**Strengths:**
- No AWS credentials needed (S3 proxy pattern)
- ComfyUI API key support (Bearer token)
- No secrets in source code
- `.env` file for configuration

**Weaknesses:**
- RunPod API key passed as URL query parameter in GraphQL call
- No input sanitization on prompt text before passing to ComfyUI
- Error messages truncated to 2000 chars but not sanitized

---

## Risk Summary

| Risk | Severity | Detail |
|------|----------|--------|
| No tests | CRITICAL | Any change could break the execution pipeline undetected |
| Unpinned dependencies | CRITICAL | `pip install` could pull incompatible versions at any time |
| No packaging | HIGH | Cannot install as a package or manage versions |
| Untested faceswap/LoRA paths | HIGH | Workflow code paths for ReActor, FaceFusion, and user LoRAs are untested |
| In-memory LoRA download | MEDIUM | Large files loaded entirely into memory (hundreds of MB) |
| No structured logging | MEDIUM | Harder to parse logs in production |
| Hardcoded node IDs | MEDIUM | Fragile coupling to specific ComfyUI workflow structure |

---

## Recommended Priority Actions

1. **Add dependency pinning** -- Generate `requirements.lock` or use `pip-tools`
2. **Add pyproject.toml** -- Enable proper packaging and tooling
3. **Write core tests** -- At minimum: workflow_builder, schemas validation, executor mocking
4. **Add type annotations** -- Especially to main.py loop functions and registry_client.py
5. **Test faceswap + LoRA paths end-to-end** -- These are built but completely untested
6. **Consider streaming downloads** -- For LoRA files to reduce memory pressure
