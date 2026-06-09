# Add Moment Search as the Primary Search API

The primary benchmark and Search Interface contract is a new Moment Search API that accepts a selected media ID, Semantic Query, Top-K, and Evaluation Profile, then returns ranked Video Moments with scores and Evidence. The older episodic search endpoint can remain for compatibility, but it should not define the benchmark path because it exposes audio-centric chunks and optional LLM reasoning.
