# Architecture, Frontend, and Backend Audit

## Verified Architecture

```text
Next.js Studio
  -> FastAPI API routes
  -> HybridRecommendationEngine
       -> processed Parquet catalog/interactions/playlists
       -> NeuMF + track encoder + causal transformer checkpoints
       -> exact NumPy/BLAS inner-product index
       -> item-KNN state
       -> live per-user feedback state
       -> adaptive/fisher/artist-spaced shuffle engines
```

## Findings

### High

1. **Missing model artifacts previously allowed silent fallback state.**
   - If processed Parquet files existed but model artifacts did not, the engine could construct untrained NeuMF/Transformer objects and export fallback embeddings.
   - The engine now requires NeuMF, transformer, track embedding, and item-CF artifacts before startup.

### Medium

2. **CORS was configured with wildcard origins and credentials.**
   - The API now defaults to localhost frontend origins and accepts `ALLOWED_ORIGINS` as a comma-separated environment setting.

3. **FastAPI and UI provide no authentication or authorization.**
   - This is appropriate for a local demo but unsafe for an internet-facing deployment.

4. **Feedback state is process-memory only.**
   - Feedback and LinUCB adaptation disappear on backend restart and are not shared across workers.

5. **The UI loading copy claimed FAISS while runtime code uses exact NumPy/BLAS similarity search.**
   - Source and UI text are now corrected to describe exact inner-product retrieval.

## Frontend Mapping

- The UI has typed API wrappers and tabs for discovery, shuffle, search, feedback, and benchmarks.
- It sends live feedback and refreshes recommendations/profiles.
- It displays the checked-in benchmark JSON from the backend.
- It has no frontend test or lint script.
- It assumes a localhost backend unless `NEXT_PUBLIC_API_URL` is set.

## Applied Fixes

- Restricted CORS origins with an environment override.
- Added explicit missing-artifact startup validation.
- API lifespan catches incomplete artifact state, leaves the engine unavailable, and keeps `/api/health` reachable as a clear 503 instead of crashing startup.
- Corrected source and UI text that claimed FAISS.
- Simplified the manual horizon override and added a focused test.
- Relaxed brittle API test assumptions about exact catalog size and user count.
