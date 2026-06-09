import { useCallback, useRef, useState } from "react";
import apiClient from "../api/client";

const PAGE_SIZE = 5;

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

function ResultCard({ moment, onSeek }) {
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
        onClick={() => onSeek(moment.start_sec)}
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
  const [mediaId, setMediaId] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [durationSec, setDurationSec] = useState(null);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [profile, setProfile] = useState("activitynet_visual_heavy");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [page, setPage] = useState(0);

  const handleSearch = useCallback(
    async (e) => {
      e.preventDefault();
      if (!mediaId.trim() || query.trim().length < 3) return;

      setLoading(true);
      setError(null);
      setResponse(null);
      setPage(0);

      try {
        const { data } = await apiClient.post("/api/search/moments", {
          media_id: mediaId.trim(),
          query: query.trim(),
          top_k: topK,
          duration_sec: durationSec,
          profile,
        });
        setResponse(data);
      } catch (err) {
        const detail = err.response?.data?.detail ?? err.message;
        setError(typeof detail === "string" ? detail : JSON.stringify(detail));
      } finally {
        setLoading(false);
      }
    },
    [durationSec, mediaId, profile, query, topK],
  );

  const handleSeek = useCallback((sec) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = sec;
    videoRef.current.play?.().catch(() => {});
  }, []);

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

      <div className="mb-8 grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.8fr)]">
        <section>
          <div className="aspect-video overflow-hidden rounded-lg border border-slate-800 bg-black">
            {videoUrl.trim() ? (
              <video
                ref={videoRef}
                controls
                preload="metadata"
                src={videoUrl.trim()}
                onLoadedMetadata={(e) => setDurationSec(e.currentTarget.duration)}
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
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              required
            />
          </div>

          <div>
            <label htmlFor="videoUrl" className="mb-1 block text-sm font-medium text-slate-400">
              Video URL
            </label>
            <input
              id="videoUrl"
              type="url"
              value={videoUrl}
              onChange={(e) => {
                setVideoUrl(e.target.value);
                setDurationSec(null);
              }}
              placeholder="https://..."
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
                <option value="activitynet_visual_heavy">ActivityNet</option>
                <option value="castle_lifelog_balanced">CASTLE</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !mediaId.trim() || query.trim().length < 3}
            className="w-full rounded-lg bg-emerald-600 px-4 py-3 font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Searching..." : "Search Moments"}
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
          <p className="mb-4 text-sm text-slate-400">
            {response.total} result{response.total !== 1 ? "s" : ""} for "{response.query}"
            <span className="ml-2 text-slate-500">({response.profile})</span>
          </p>

          {results.length === 0 ? (
            <p className="text-center text-slate-500">No matching video moments found.</p>
          ) : (
            <>
              <div className="space-y-4">
                {pageResults.map((moment) => (
                  <ResultCard key={moment.moment_id} moment={moment} onSeek={handleSeek} />
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
