# Style Guide

Functions use snake_case, types use PascalCase, and constants, enum values, and fault names use SCREAMING_SNAKE_CASE. This follows C3 convention throughout the codebase.

Panics and asserts are reserved for impossible or unrecoverable states that indicate internal bugs. Recoverable errors are returned to the caller as optionals with named faults. Faults are strictly for reporting invalid conditions and must never be used for positive control flow. User-facing API names should be brief and concise, while internal implementation names should be verbose and descriptive of exactly what they do.

Device-level details must not leak into the graph API, and graph-specific concerns must not leak into the tensor API. Allocation should be kept off the hot path. One-time or cached processes may use dynamic memory freely, but hot paths should prefer arenas, fixed buffers, or free lists. Fixed buffers are preferred when there is a natural upper bound on size.
