# Sample Visual Evidence Every Two Seconds

The baseline visual pipeline samples one keyframe every two seconds and aggregates frame-level visual scores into each 10-second Video Moment using the maximum frame score within the window. This reduces the chance of missing short visible actions while keeping indexing cost manageable for the first ActivityNet validation pass.
