import { useCallback, useState } from "react";
import apiClient from "../api/client";

const PAGE_SIZE = 5;

function formatTimestamp(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-xl border border-slate-800 bg-slate-900 p-5">
      <div className="mb-3 h-4 w-1/3 rounded bg-slate-700" />
      <div className="mb-2 h-3 w-full rounded bg-slate-800" />
      <div className="mb-4 h-3 w-2/3 rounded bg-slate-800" />
      <div className="h-10 w-full rounded bg-slate-800" />
    </div>
  );
}

function ResultCard({ hit, rank }) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-900 p-5 shadow-lg">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
            #{rank}
          </span>
          <h3 className="mt-1 font-mono text-sm text-slate-300">{hit.chunk_id}</h3>
        </div>
        <span className="rounded-full bg-emerald-900/50 px-3 py-1 text-sm font-semibold text-emerald-400">
          {(hit.score * 100).toFixed(1)}%
        </span>
      </div>

      <dl className="mb-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
        <div>
          <dt className="inline">Media </dt>
          <dd className="inline font-mono text-slate-300">{hit.media_id}</dd>
        </div>
        <div>
          <dt className="inline">Time </dt>
          <dd className="inline text-slate-300">
            {formatTimestamp(hit.start_sec)} – {formatTimestamp(hit.end_sec)}
          </dd>
        </div>
        <div>
          <dt className="inline">Duration </dt>
          <dd className="inline text-slate-300">{hit.duration_sec.toFixed(1)}s</dd>
        </div>
        <div>
          <dt className="inline">Lang </dt>
          <dd className="inline text-slate-300">{hit.language}</dd>
        </div>
      </dl>

      {hit.transcript && (
        <p className="mb-4 text-sm leading-relaxed text-slate-300">{hit.transcript}</p>
      )}

      {hit.minio_url && (
        <audio controls preload="none" className="w-full" src={hit.minio_url}>
          Your browser does not support the audio element.
        </audio>
      )}
    </article>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [scoreThreshold, setScoreThreshold] = useState(0);
  const [useLlm, setUseLlm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [page, setPage] = useState(0);

  const handleSearch = useCallback(
    async (e) => {
      e.preventDefault();
      if (!query.trim()) return;

      setLoading(true);
      setError(null);
      setResponse(null);
      setPage(0);

      try {
        const { data } = await apiClient.post("/api/search/episodic", {
          query: query.trim(),
          top_k: topK,
          score_threshold: scoreThreshold > 0 ? scoreThreshold : null,
          use_llm: useLlm,
          embedder: "clap",
        });
        setResponse(data);
      } catch (err) {
        const detail = err.response?.data?.detail ?? err.message;
        setError(typeof detail === "string" ? detail : JSON.stringify(detail));
      } finally {
        setLoading(false);
      }
    },
    [query, topK, scoreThreshold, useLlm],
  );

  const hits = response?.hits ?? [];
  const totalPages = Math.ceil(hits.length / PAGE_SIZE);
  const pageHits = hits.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="mx-auto min-h-screen max-w-3xl px-4 py-10">
      <header className="mb-10 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          NTCIR CSAT Episodic Search
        </h1>
        <p className="mt-2 text-slate-400">
          Natural-language search over indexed lifelog audio memory
        </p>
      </header>

      <form onSubmit={handleSearch} className="mb-8 space-y-4">
        <div>
          <label htmlFor="query" className="mb-1 block text-sm font-medium text-slate-400">
            Query
          </label>
          <input
            id="query"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. eating lunch at the cafeteria"
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-white placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            minLength={3}
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <div>
            <label htmlFor="topK" className="mb-1 block text-sm text-slate-400">
              Top-K
            </label>
            <input
              id="topK"
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
            />
          </div>
          <div>
            <label htmlFor="threshold" className="mb-1 block text-sm text-slate-400">
              Min score
            </label>
            <input
              id="threshold"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={scoreThreshold}
              onChange={(e) => setScoreThreshold(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-white"
            />
          </div>
          <div className="flex items-end">
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-400">
              <input
                type="checkbox"
                checked={useLlm}
                onChange={(e) => setUseLlm(e.target.checked)}
                className="rounded border-slate-600 bg-slate-900 text-emerald-500"
              />
              LLM reasoning
            </label>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || query.trim().length < 3}
          className="w-full rounded-lg bg-emerald-600 px-4 py-3 font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

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
            {response.total_hits} result{response.total_hits !== 1 ? "s" : ""} for &ldquo;
            {response.query}&rdquo;
            {response.embedder_used && (
              <span className="ml-2 text-slate-500">({response.embedder_used})</span>
            )}
          </p>

          {response.reasoning && (
            <div className="mb-6 rounded-xl border border-indigo-800 bg-indigo-950/40 p-5">
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-indigo-400">
                LLM Reasoning
              </h2>
              <p className="mb-2 text-white">{response.reasoning.answer}</p>
              <p className="text-sm text-slate-400">{response.reasoning.reasoning}</p>
            </div>
          )}

          {hits.length === 0 ? (
            <p className="text-center text-slate-500">No matching audio chunks found.</p>
          ) : (
            <>
              <div className="space-y-4">
                {pageHits.map((hit, i) => (
                  <ResultCard key={hit.chunk_id} hit={hit} rank={page * PAGE_SIZE + i + 1} />
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
