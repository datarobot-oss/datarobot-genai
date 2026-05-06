# NAT integration tests (legacy DRUM path)

These tests exercise NAT end-to-end using the legacy integration path (the way NAT works with DRUM).

## Run

1. Copy values from `.env.sample` into your root `.env` file.
2. Set `MEM0_API_KEY` to run the real Mem0 auto-memory wrapper test. Without it, that test is skipped.
3. Run `task test-nat-integration`.
