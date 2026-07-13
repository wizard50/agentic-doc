## What Is Ownership?

Ownership is Rust's system for managing memory without a garbage collector.
Each value has a single owner, and when the owner goes out of scope the value is dropped.

## Moving Values

When ownership is transferred to another variable, the original binding can no longer be used.