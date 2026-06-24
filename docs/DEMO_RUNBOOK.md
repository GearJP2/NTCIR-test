# Demo Runbook

## Goal

Use the web demo to show ActivityNet controlled-proxy moment retrieval:

- search with a natural-language event query
- retrieve ranked timestamp windows across a small ActivityNet sample collection
- click `Seek` to load the matched video and jump to the retrieved moment

## Start

Terminal 1:

```bash
make dev
```

Terminal 2:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

Open:

```text
http://localhost:5173/
```

## Demo Script

Use `Collection` mode first. This mode searches the manifest-backed indexed set, not only the visible query preset buttons.

- Query: `The person spreads a jam on the cake.`
- Profile: `ActivityNet visual only`
- Collection Manifest: `data/manifests/activitynet_dev200_indexed_current.jsonl`
- Candidate Hits: `1000`
- Window: `10`
- Stride: `5`

Expected first result:

- `v_vopKTwCiHrA:45.000-55.000`
- This matches the ActivityNet ground-truth segment around `44.55-54.27s`.

Click `Seek` on the first result to load the cake video and jump to the retrieved moment.

Use `Single video` mode when explaining the benchmark metric:

- ActivityNet quantitative evaluation is measured inside a selected video.
- Collection mode is a real collection-level retrieval demo over the indexed ActivityNet manifest.
- The retrieved moment contract is still the same: query plus media ID plus timestamp window.

Use `Evaluation` mode when checking correctness against ActivityNet ground truth:

- Collection Manifest: `data/manifests/activitynet_dev200_indexed_current.jsonl`
- Select an Evaluation Query from the dropdown.
- Click `Evaluate Query`.
- The page shows the ground-truth interval, best tIoU, Hit Rank, and per-result tIoU/Hit labels.

Known sanity-check query:

- Query ID: `v_vopKTwCiHrA:1`
- Ground truth: `44.55-54.27s`
- Expected first result: `v_vopKTwCiHrA:45.000-55.000`
- Expected tIoU: about `0.887`

## Notes

- The demo needs Docker/Milvus running and ActivityNet videos indexed.
- The frontend proxies `/api` and `/media` to the FastAPI backend on port `8000`.
- `/media/activitynet/...` is served only when `data/activitynet/videos` exists locally.
- For the report, frame this as a temporal-grounding demo on ActivityNet, not as a CASTLE quantitative result.
