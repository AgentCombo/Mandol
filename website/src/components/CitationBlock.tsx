import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/translations';
import { copyText } from '@site/src/utils/copyText';

const bibtex = `@misc{zhang2026mandol,
  title={Mandol: An Agglomerative Agent Memory System for Long-Term Conversations},
  author={Yuhan Zhang and Zhiyuan Guo and Ziheng Zeng and Wei Wang and Wentao Wu and Lijie Xu},
  year={2026},
  eprint={2606.29778},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  doi={10.48550/arXiv.2606.29778},
  url={https://arxiv.org/abs/2606.29778}
}`;

export default function CitationBlock(): React.JSX.Element {
  const [copied, setCopied] = useState(false);
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.15 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const handleCopy = async () => {
    if (!(await copyText(bibtex))) return;
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section ref={ref} className="section-darker py-24 sm:py-32">
      <div className="mx-auto max-w-3xl px-6">
        <div className="mb-10 text-center">
          <h2
            className={`site-heading text-3xl font-bold tracking-tight sm:text-4xl animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
          >
            {t.citationTitle}
          </h2>
          <p
            className={`site-text-subtle mt-3 text-sm animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
            style={{ animationDelay: '0.1s' }}
          >
            {t.citationSubtitle}
          </p>
        </div>

        <div
          className={`code-window animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.2s' }}
        >
          <div className="code-window-bar">
            <div className="code-dot code-dot-red" />
            <div className="code-dot code-dot-yellow" />
            <div className="code-dot code-dot-green" />
            <div className="flex-1 text-center text-[11px] text-white/20 font-mono">
              mandol.bib
            </div>
            <button
              type="button"
              onClick={handleCopy}
              className={`copy-btn ${copied ? 'copied' : ''}`}
              aria-label={copied ? t.citationCopied : t.citationCopy}
            >
              {copied ? (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  {t.citationCopied}
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  {t.citationCopy}
                </>
              )}
            </button>
          </div>
          <pre className="code-content bibtex-content"><code>{bibtex}</code></pre>
        </div>

        <p
          className={`site-text-faint mt-5 text-center text-[12px] animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.35s' }}
        >
          {t.citationNote}
        </p>
      </div>
    </section>
  );
}
