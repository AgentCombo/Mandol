import React from 'react';

const innovations = [
  {
    title: 'Hierarchical Theoretical Memory Model',
    points: [
      'Foundation memory layer — structured semantic graph for unified representation and traceable associations of raw information',
      'High-level memory layer — automatic pattern mining and induction to distill traceable abstract memories',
      'Intelligent query layer — query-adaptive routing and quantitative retrieval across both memory layers',
    ],
  },
  {
    title: 'Unified In-Memory Storage Architecture',
    points: [
      'In-memory semantic data structures that fuse CRUD, multi-modal vector search, and structured graph capabilities in a single atomic store',
      'Co-designed SemanticMap + SemanticGraph eliminate cross-store I/O and provide unified hybrid retrieval interfaces',
      'Extended to persistence for an "active-memory / durable-storage" synergy balancing real-time performance and capacity',
    ],
  },
  {
    title: 'Intelligent Routing & Quantitative Retrieval',
    points: [
      'Query-adaptive routing with token budget allocation and multi-source parallel recall across foundation and high-level layers',
      'Two-stage quantitative denoising and conflict resolution filters noise and resolves contradictory signals',
      'Context generation optimized under token constraints without invoking LLMs during retrieval',
    ],
  },
];

export default function FeaturesGrid(): JSX.Element {
  return (
    <section className="section-dark py-24">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            What is <span className="gradient-text">Mandol</span>?
          </h2>
          {/* <p className="mt-4 text-lg text-white/50">
            An in-memory agent memory system built on three key innovations.
          </p> */}
        </div>

        {/* System overview image */}
        <div className="mb-16 flex justify-center">
          <img
            src="/Mandol/img/mandol-overview.png"
            alt="Mandol System Overview"
            className="w-full max-w-4xl rounded-xl"
          />
        </div>

        {/* Three innovations — concise cards */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          {innovations.map((inv) => (
            <div key={inv.title} className="card-highlight">
              <h3 className="text-lg font-semibold text-primary-300">
                {inv.title}
              </h3>
              <ul className="mt-4 space-y-3">
                {inv.points.map((p) => (
                  <li key={p} className="flex items-start gap-2 text-sm leading-relaxed text-white/65">
                    <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-500/60" />
                    {p}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}