# Use Fixed Windows as the Baseline Video Moment Unit

The first retrieval baseline represents Video Moments as 10-second windows with a 5-second stride. This keeps ingestion, search, and temporal evaluation simple while the project validates semantic retrieval quality with Recall@10 at tIoU >= 0.3; dynamic boundary refinement can be added later if fixed windows limit evaluation scores.
