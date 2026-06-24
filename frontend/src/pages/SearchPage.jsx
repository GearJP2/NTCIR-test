import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import apiClient from "../api/client";

const PAGE_SIZE = 5;
const DEMO_SAMPLES = [
  {
    label: "Pumpkin cake",
    mediaId: "v_vopKTwCiHrA",
    videoUrl: "/media/activitynet/v_vopKTwCiHrA.mp4",
    durationSec: 162.01,
    query: "The person spreads a jam on the cake.",
  },
  {
    label: "Drum kit",
    mediaId: "v_LN8UWHvoELs",
    videoUrl: "/media/activitynet/v_LN8UWHvoELs.mp4",
    durationSec: 202.83,
    query: "The man begins playing the drum with his sticks.",
  },
  {
    label: "Sit ups",
    mediaId: "v_RzMKERQ9vOU",
    videoUrl: "/media/activitynet/v_RzMKERQ9vOU.mp4",
    durationSec: 157.01,
    query: "The man performs sit ups on the ground.",
  },
  {
    label: "Field game",
    mediaId: "v_a74RMGL_c8E",
    videoUrl: "/media/activitynet/v_a74RMGL_c8E.mp4",
    durationSec: 32.0,
    query: "A player hits the ball into the goal.",
  },
];

const PROFILES = [
  ["activitynet_visual_only", "ActivityNet visual only"],
  ["activitynet_visual_asr_light", "ActivityNet visual + ASR light"],
  ["activitynet_visual_audio_light", "ActivityNet visual + audio light"],
  ["activitynet_visual_heavy", "ActivityNet heavy"],
  ["castle_lifelog_balanced", "CASTLE lifelog balanced"],
];
const DEFAULT_COLLECTION_MANIFEST = "data/manifests/activitynet_dev200_indexed_current.jsonl";

function formatTimestamp(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function formatScore(score) {
  return `${(score * 100).toFixed(1)}%`;
}

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="mb-3 h-4 w-1/3 rounded bg-slate-700" />
      <div className="mb-2 h-3 w-full rounded bg-slate-800" />
      <div className="mb-4 h-3 w-2/3 rounded bg-slate-800" />
      <div className="h-10 w-full rounded bg-slate-800" />
    </div>
  );
}

function EvidenceList({ evidence }) {
  if (!evidence?.length) return null;

  return (
    <div className="mt-4 flex flex-wrap gap-2">
      {evidence.map((item, index) => (
        <span
          key={`${item.source_type}-${item.source_id ?? index}`}
          className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-300"
        >
          {item.source_type} {formatScore(item.score)}
        </span>
      ))}
    </div>
  );
}

function ResultCard({ moment, onSeek, evaluation }) {
  const duration = moment.end_sec - moment.start_sec;

  return (
    <article className="rounded-lg border border-slate-800 bg-slate-900 p-5 shadow-lg">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="text-xs font-medium uppercase text-slate-500">
            #{moment.rank}
          </span>
          <h3 className="mt-1 truncate font-mono text-sm text-slate-300">
            {moment.moment_id}
          </h3>
        </div>
        <span className="rounded-full bg-emerald-900/50 px-3 py-1 text-sm font-semibold text-emerald-400">
          {formatScore(moment.score)}
        </span>
      </div>

      {evaluation && (
        <div
          className={`mb-4 rounded-lg border px-3 py-2 text-sm ${
            evaluation.hit
              ? "border-emerald-800 bg-emerald-950/40 text-emerald-300"
              : "border-slate-700 bg-slate-950 text-slate-300"
          }`}
        >
          tIoU {formatScore(evaluation.tiou)}
          <span className="ml-2 text-xs uppercase">
            {evaluation.hit ? "Hit" : "Miss"}
          </span>
        </div>
      )}

      <dl className="mb-4 grid grid-cols-1 gap-x-4 gap-y-1 text-xs text-slate-400 sm:grid-cols-2">
        <div>
          <dt className="inline">Media </dt>
          <dd className="inline font-mono text-slate-300">{moment.media_id}</dd>
        </div>
        <div>
          <dt className="inline">Time </dt>
          <dd className="inline text-slate-300">
            {formatTimestamp(moment.start_sec)} - {formatTimestamp(moment.end_sec)}
          </dd>
        </div>
        <div>
          <dt className="inline">Duration </dt>
          <dd className="inline text-slate-300">{duration.toFixed(1)}s</dd>
        </div>
        <div>
          <dt className="inline">Thumbnail </dt>
          <dd className="inline text-slate-300">{formatTimestamp(moment.thumbnail_sec)}</dd>
        </div>
      </dl>

      <button
        type="button"
        onClick={() => onSeek(moment)}
        className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500"
      >
        Seek
      </button>

      <EvidenceList evidence={moment.evidence} />
    </article>
  );
}

export default function SearchPage() {
  const videoRef = useRef(null);
  const pendingSeekRef = useRef(null);
  const [searchMode, setSearchMode] = useState("collection");
  const [mediaId, setMediaId] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [durationSec, setDurationSec] = useState(null);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [windowSec, setWindowSec] = useState(10);
  const [strideSec, setStrideSec] = useState(5);
  const [manifestPath, setManifestPath] = useState(DEFAULT_COLLECTION_MANIFEST);
  const [candidateLimit, setCandidateLimit] = useState(1000);
  const [evalQueries, setEvalQueries] = useState([]);
  const [evalQueryId, setEvalQueryId] = useState("");
  const [evalResponse, setEvalResponse] = useState(null);
  const [profile, setProfile] = useState("activitynet_visual_only");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [page, setPage] = useState(0);

  const runSingleVideoSearch = useCallback(
    async ({ targetMediaId, targetDurationSec, targetQuery, perVideoTopK }) => {
      const { data } = await apiClient.post("/api/search/moments", {
        media_id: targetMediaId,
        query: targetQuery,
        top_k: perVideoTopK,
        duration_sec: targetDurationSec,
        window_sec: windowSec,
        stride_sec: strideSec,
        profile,
      });
      return data;
    },
    [profile, strideSec, windowSec],
  );

  const handleSearch = useCallback(
    async (e) => {
      e.preventDefault();
      if (query.trim().length < 3) return;
      if (searchMode === "single" && (!mediaId.trim() || !durationSec)) return;

      setLoading(true);
      setError(null);
      setResponse(null);
      setEvalResponse(null);
      setPage(0);

      try {
        if (searchMode === "evaluation") {
          const { data } = await apiClient.post("/api/search/moments/evaluate", {
            query_id: evalQueryId,
            top_k: topK,
            manifest_path: manifestPath.trim() || DEFAULT_COLLECTION_MANIFEST,
            window_sec: windowSec,
            stride_sec: strideSec,
            profile,
            tiou_threshold: 0.3,
          });
          setResponse(data.search_response);
          setEvalResponse(data);
          setMediaId(data.media_id);
          setVideoUrl(`/media/activitynet/${data.media_id}.mp4`);
          setDurationSec(null);
          setQuery(data.query);
        } else if (searchMode === "collection") {
          const { data } = await apiClient.post("/api/search/moments/collection", {
            query: query.trim(),
            top_k: topK,
            manifest_path: manifestPath.trim() || DEFAULT_COLLECTION_MANIFEST,
            candidate_limit: candidateLimit,
            window_sec: windowSec,
            stride_sec: strideSec,
            profile,
          });
          setResponse(data);
        } else {
          const data = await runSingleVideoSearch({
            targetMediaId: mediaId.trim(),
            targetDurationSec: durationSec,
            targetQuery: query.trim(),
            perVideoTopK: topK,
          });
          setResponse(data);
        }
      } catch (err) {
        const detail = err.response?.data?.detail ?? err.message;
        setError(typeof detail === "string" ? detail : JSON.stringify(detail));
      } finally {
        setLoading(false);
      }
    },
    [
      candidateLimit,
      durationSec,
      evalQueryId,
      manifestPath,
      mediaId,
      profile,
      query,
      runSingleVideoSearch,
      searchMode,
      strideSec,
      topK,
      windowSec,
    ],
  );

  const applySample = useCallback((sample) => {
    setMediaId(sample.mediaId);
    setVideoUrl(sample.videoUrl);
    setQuery(sample.query);
    setDurationSec(null);
    setResponse(null);
    setEvalResponse(null);
    setError(null);
    setPage(0);
  }, []);

  const seekCurrentVideo = useCallback((sec) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = sec;
    video.play?.().catch(() => {});
  }, []);

  const handleVideoMetadata = useCallback(
    (e) => {
      setDurationSec(e.currentTarget.duration);
      if (pendingSeekRef.current !== null) {
        seekCurrentVideo(pendingSeekRef.current);
        pendingSeekRef.current = null;
      }
    },
    [seekCurrentVideo],
  );

  const handleSeek = useCallback((moment) => {
    const sample = DEMO_SAMPLES.find((item) => item.mediaId === moment.media_id);
    const nextVideoUrl = sample?.videoUrl ?? `/media/activitynet/${moment.media_id}.mp4`;
    if (nextVideoUrl !== videoUrl) {
      pendingSeekRef.current = moment.start_sec;
      setMediaId(moment.media_id);
      setVideoUrl(nextVideoUrl);
      setDurationSec(null);
      return;
    }
    seekCurrentVideo(moment.start_sec);
  }, [seekCurrentVideo, videoUrl]);

  const canSearch = (
    searchMode === "evaluation"
      ? Boolean(evalQueryId)
      : query.trim().length >= 3 && (
        searchMode === "collection" || (mediaId.trim() && durationSec)
      )
  );

  const resultScopeLabel = searchMode === "collection" ? response?.media_id : mediaId;

  const handleModeChange = useCallback((mode) => {
    setSearchMode(mode);
    setResponse(null);
    setEvalResponse(null);
    setError(null);
    setPage(0);
  }, []);

  useEffect(() => {
    if (searchMode !== "evaluation") return;

    let active = true;
    apiClient
      .get("/api/search/moments/evaluation-queries", {
        params: {
          manifest_path: manifestPath.trim() || DEFAULT_COLLECTION_MANIFEST,
          limit: 500,
        },
      })
      .then(({ data }) => {
        if (!active) return;
        setEvalQueries(data.queries);
        setEvalQueryId((current) => {
          const selected = data.queries.find((item) => item.query_id === current) ?? data.queries[0];
          if (selected) {
            setQuery(selected.query);
            setMediaId(selected.media_id);
            setVideoUrl(`/media/activitynet/${selected.media_id}.mp4`);
            setDurationSec(null);
          }
          return selected?.query_id || "";
        });
      })
      .catch((err) => {
        if (!active) return;
        const detail = err.response?.data?.detail ?? err.message;
        setError(typeof detail === "string" ? detail : JSON.stringify(detail));
      });

    return () => {
      active = false;
    };
  }, [manifestPath, searchMode]);

  const selectedEvalQuery = useMemo(
    () => evalQueries.find((item) => item.query_id === evalQueryId),
    [evalQueries, evalQueryId],
  );

  const evaluationByMomentId = useMemo(() => {
    const items = evalResponse?.evaluated_results ?? [];
    return Object.fromEntries(items.map((item) => [item.moment.moment_id, item]));
  }, [evalResponse]);

  const results = response?.results ?? [];
  const totalPages = Math.ceil(results.length / PAGE_SIZE);
  const pageResults = results.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="mx-auto min-h-screen max-w-5xl px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          NTCIR CSAT Moment Search
        </h1>
        <p className="mt-2 text-slate-400">
          Single-video semantic search returning ranked timestamped moments.
        </p>
      </header>

      <section className="mb-6">
        <div className="flex flex-wrap gap-2">
          {DEMO_SAMPLES.map((sample) => (
            <button
              key={sample.mediaId}
              type="button"
              onClick={() => applySample(sample)}
              className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 transition hover:border-emerald-500 hover:text-white"
            >
              {sample.label}
            </button>
          ))}
        </div>
      </section>

      <div className="mb-8 grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.8fr)]">
        <section>
          <div className="aspect-video overflow-hidden rounded-lg border border-slate-800 bg-black">
            {videoUrl.trim() ? (
              <video
                ref={videoRef}
                controls
                preload="metadata"
                src={videoUrl.trim()}
                onLoadedMetadata={handleVideoMetadata}
                className="h-full w-full"
              />
            ) : (
              <div className="flex h-full items-center justify-center px-4 text-center text-sm text-slate-500">
                No video selected
              </div>
            )}
          </div>
        </section>

        <form onSubmit={handleSearch} className="space-y-4">
          <div className="grid grid-cols-3 gap-2 rounded-lg border border-slate-800 bg-slate-950 p-1">
            <button
              type="button"
              onClick={() => handleModeChange("collection")}
              className={`rounded-md px-3 py-2 text-sm font-medium transition ${
                searchMode === "collection"
                  ? "bg-emerald-600 text-white"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Collection
            </button>
            <button
              type="button"
              onClick={() => handleModeChange("evaluation")}
              className={`rounded-md px-3 py-2 text-sm font-medium transition ${
                searchMode === "evaluation"
                  ? "bg-emerald-600 text-white"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Evaluation
            </button>
            <button
              type="button"
              onClick={() => handleModeChange("single")}
              className={`rounded-md px-3 py-2 text-sm font-medium transition ${
                searchMode === "single"
                  ? "bg-emerald-600 text-white"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Single video
            </button>
          </div>

          {(searchMode === "collection" || searchMode === "evaluation") && (
            <div>
              <label htmlFor="manifestPath" className="mb-1 block text-sm font-medium text-slate-400">
                Collection Manifest
              </label>
              <input
                id="manifestPath"
                type="text"
                value={manifestPath}
                onChange={(e) => setManifestPath(e.target.value)}
                placeholder={DEFAULT_COLLECTION_MANIFEST}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>
          )}

          {searchMode === "evaluation" && (
            <div>
              <label htmlFor="evalQuery" className="mb-1 block text-sm font-medium text-slate-400">
                Evaluation Query
              </label>
              <select
                id="evalQuery"
                value={evalQueryId}
                onChange={(e) => {
                  const next = e.target.value;
                  setEvalQueryId(next);
                  const selected = evalQueries.find((item) => item.query_id === next);
                  if (selected) {
                    setQuery(selected.query);
                    setMediaId(selected.media_id);
                    setVideoUrl(`/media/activitynet/${selected.media_id}.mp4`);
                    setDurationSec(null);
                  }
                }}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
              >
                {evalQueries.map((item) => (
                  <option key={item.query_id} value={item.query_id}>
                    {item.query_id} | {item.query}
                  </option>
                ))}
              </select>
              {selectedEvalQuery && (
                <p className="mt-2 text-xs text-slate-400">
                  GT {formatTimestamp(selectedEvalQuery.ground_truth.start_sec)} - {formatTimestamp(selectedEvalQuery.ground_truth.end_sec)}
                </p>
              )}
            </div>
          )}

          <div>
            <label htmlFor="mediaId" className="mb-1 block text-sm font-medium text-slate-400">
              Media ID
            </label>
            <input
              id="mediaId"
              type="text"
              value={mediaId}
              onChange={(e) => setMediaId(e.target.value)}
              placeholder="v_123"
              disabled={searchMode === "collection" || searchMode === "evaluation"}
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              required={searchMode === "single"}
            />
          </div>

          <div>
            <label htmlFor="videoUrl" className="mb-1 block text-sm font-medium text-slate-400">
              Video URL
            </label>
            <input
              id="videoUrl"
              type="text"
              value={videoUrl}
              onChange={(e) => {
                setVideoUrl(e.target.value);
                setDurationSec(null);
              }}
              placeholder="/media/activitynet/v_123.mp4"
              disabled={searchMode === "collection" || searchMode === "evaluation"}
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>

          <div>
            <label htmlFor="query" className="mb-1 block text-sm font-medium text-slate-400">
              Semantic Query
            </label>
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="woman doing sit ups"
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              minLength={3}
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="topK" className="mb-1 block text-sm text-slate-400">
                Top-K
              </label>
              <input
                id="topK"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
              />
            </div>
            <div>
              <label htmlFor="profile" className="mb-1 block text-sm text-slate-400">
                Profile
              </label>
              <select
                id="profile"
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
              >
                {PROFILES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="windowSec" className="mb-1 block text-sm text-slate-400">
                Window (s)
              </label>
              <input
                id="windowSec"
                type="number"
                min={1}
                step={1}
                value={windowSec}
                onChange={(e) => setWindowSec(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
              />
            </div>
            <div>
              <label htmlFor="strideSec" className="mb-1 block text-sm text-slate-400">
                Stride (s)
              </label>
              <input
                id="strideSec"
                type="number"
                min={1}
                step={1}
                value={strideSec}
                onChange={(e) => setStrideSec(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
              />
            </div>
          </div>

          {searchMode === "collection" && (
            <div>
              <label htmlFor="candidateLimit" className="mb-1 block text-sm text-slate-400">
                Candidate Hits
              </label>
              <input
                id="candidateLimit"
                type="number"
                min={1}
                max={5000}
                step={50}
                value={candidateLimit}
                onChange={(e) => setCandidateLimit(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
              />
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !canSearch}
            className="w-full rounded-lg bg-emerald-600 px-4 py-3 font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? "Searching..."
              : searchMode === "collection"
                ? "Search Collection"
                : searchMode === "evaluation"
                  ? "Evaluate Query"
                  : "Search Moments"}
          </button>
        </form>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-red-800 bg-red-950/50 px-4 py-3 text-red-300">
          {error}
        </div>
      )}

      {loading && (
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      )}

      {response && !loading && (
        <section>
          {evalResponse && (
            <div className="mb-6 rounded-lg border border-slate-800 bg-slate-900 p-4">
              <div className="grid gap-3 text-sm text-slate-300 sm:grid-cols-4">
                <div>
                  <span className="block text-xs uppercase text-slate-500">Ground Truth</span>
                  {formatTimestamp(evalResponse.ground_truth.start_sec)} - {formatTimestamp(evalResponse.ground_truth.end_sec)}
                </div>
                <div>
                  <span className="block text-xs uppercase text-slate-500">Best tIoU</span>
                  {formatScore(evalResponse.best_tiou)}
                </div>
                <div>
                  <span className="block text-xs uppercase text-slate-500">Threshold</span>
                  {formatScore(evalResponse.tiou_threshold)}
                </div>
                <div>
                  <span className="block text-xs uppercase text-slate-500">Hit Rank</span>
                  {evalResponse.hit_rank ?? "Miss"}
                </div>
              </div>
            </div>
          )}

          <p className="mb-4 text-sm text-slate-400">
            {response.total} result{response.total !== 1 ? "s" : ""} for "{response.query}"
            <span className="ml-2 text-slate-500">({response.profile}, {resultScopeLabel})</span>
          </p>

          {results.length === 0 ? (
            <p className="text-center text-slate-500">No matching video moments found.</p>
          ) : (
            <>
              <div className="space-y-4">
                {pageResults.map((moment) => (
                  <ResultCard
                    key={`${moment.rank}-${moment.moment_id}`}
                    moment={moment}
                    onSeek={handleSeek}
                    evaluation={evaluationByMomentId[moment.moment_id]}
                  />
                ))}
              </div>

              {totalPages > 1 && (
                <nav className="mt-6 flex items-center justify-center gap-4">
                  <button
                    type="button"
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="rounded-lg border border-slate-700 px-4 py-2 text-sm disabled:opacity-40"
                  >
                    Previous
                  </button>
                  <span className="text-sm text-slate-400">
                    Page {page + 1} of {totalPages}
                  </span>
                  <button
                    type="button"
                    onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                    disabled={page >= totalPages - 1}
                    className="rounded-lg border border-slate-700 px-4 py-2 text-sm disabled:opacity-40"
                  >
                    Next
                  </button>
                </nav>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
}
