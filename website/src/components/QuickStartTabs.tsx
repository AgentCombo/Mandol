import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/data/translations';

const codeSnippets: Record<string, string> = {
  install: `# Install Mandol (zero mandatory dependencies)
pip install mandol

# Optional backends
pip install mandol[faiss]      # FAISS vector index acceleration
pip install mandol[neo4j]      # Neo4j graph database
pip install mandol[all]        # Install all optional deps`,
  insert: `from mandol import MemorySystem, MemoryUnit, Uid

# Start the memory system (no-arg constructor)
system = MemorySystem.from_yaml_config("config.yaml")

# Write memories — the system auto-vectorizes text
system.add(MemoryUnit(
    uid=Uid("msg_001"),
    raw_data={"text_content": "Zhang San went to Beijing today"},
    metadata={"timestamp": "2024-01-15T10:00:00"},
))`,
  build: `# Build high-level memory structures (one call)
system.build_high_level(mode="auto")

# The system automatically performs:
#  • Session segmentation (LLM-driven)
#  • Entity & event extraction + deduplication
#  • Cross-session coreference resolution
#  • Multi-type summary & insight generation

# Query-adaptive hybrid retrieval
hits = system.holistic_retrieve(
    "Where did Zhang San go?", top_k=5
)
for hit in hits:
    print(f"[{hit.final_score:.3f}] {hit.unit.raw_data['text_content']}")`,
  save: `# One-click persistence
system.save("./memory_snapshot")

# One-click restoration
system2 = MemorySystem.load("./memory_snapshot")

# The snapshot preserves:
#  • All base & high-level memories
#  • SemanticGraph topology
#  • Vector indices & metadata`,
};

export default function QuickStartTabs(): JSX.Element {
  const [active, setActive] = useState('install');
  const [copied, setCopied] = useState(false);
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];
  const isZh = locale === 'zh-Hans';

  const tabs = [
    { id: 'install', label: t.quickStartTab1 },
    { id: 'insert', label: t.quickStartTab2 },
    { id: 'build', label: t.quickStartTab3 },
    { id: 'save', label: t.quickStartTab4 },
  ];

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const handleCopy = () => {
    navigator.clipboard.writeText(codeSnippets[active]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const code = codeSnippets[active];

  const highlightCode = (text: string) => {
    return text.split('\n').map((line, i) => {
      let className = '';
      if (line.trim().startsWith('#')) {
        className = 'comment';
      }
      return (
        <div key={i} className={className}>
          {line || ' '}
        </div>
      );
    });
  };

  return (
    <section ref={ref} className="section-dark py-24 sm:py-32">
      <div className="mx-auto max-w-3xl px-6">
        <div className="mb-12 text-center">
          <h2
            className={`text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
          >
            {isZh ? (
              <>
                快速<span className="gradient-text-blue">开始</span>
              </>
            ) : (
              <>
                Quick <span className="gradient-text-blue">Start</span>
              </>
            )}
          </h2>
          <p
            className={`mt-3 text-sm text-white/35 animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
            style={{ animationDelay: '0.1s' }}
          >
            {t.quickStartSubtitle}
          </p>
        </div>

        <div
          className={`mb-0 flex flex-wrap gap-1.5 animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.2s' }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${active === tab.id ? 'active' : ''}`}
              onClick={() => setActive(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div
          className={`code-window animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.3s' }}
        >
          <div className="code-window-bar">
            <div className="code-dot code-dot-red" />
            <div className="code-dot code-dot-yellow" />
            <div className="code-dot code-dot-green" />
            <div className="flex-1" />
            <button onClick={handleCopy} className={`copy-btn ${copied ? 'copied' : ''}`}>
              {copied ? (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  {t.quickStartCopied}
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  {t.quickStartCopy}
                </>
              )}
            </button>
          </div>
          <div className="code-content" style={{ minHeight: '200px' }}>
            {highlightCode(code)}
          </div>
        </div>

        <div
          className={`mt-8 flex items-center justify-center gap-6 text-[11px] text-white/25 animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.45s' }}
        >
          <span className={active === 'install' ? 'text-blue-400/60' : ''}>{t.quickStartStep1}</span>
          <span className="text-white/10">→</span>
          <span className={active === 'insert' ? 'text-blue-400/60' : ''}>{t.quickStartStep2}</span>
          <span className="text-white/10">→</span>
          <span className={active === 'build' ? 'text-blue-400/60' : ''}>{t.quickStartStep3}</span>
          <span className="text-white/10">→</span>
          <span className={active === 'save' ? 'text-blue-400/60' : ''}>{t.quickStartStep4}</span>
        </div>
      </div>
    </section>
  );
}
