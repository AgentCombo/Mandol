import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/data/translations';

const bibtex = `@article{mandol2026,
  title   = {Mandol: An In-Memory Layered Memory System
             for Long-Term Conversational Agents},
  author  = {Yuhan Zhang and Zhiyuan Guo and Ziheng Zeng
             and Wei Wang and Wentao Wu and Lijie Xu},
  journal = {arXiv preprint arXiv:260x.xxxxx},
  year    = {2026}
}`;

export default function CitationBlock(): JSX.Element {
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

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section ref={ref} className="section-darker py-24 sm:py-32">
      <div className="mx-auto max-w-3xl px-6">
        <div className="mb-10 text-center">
          <h2
            className={`text-3xl font-bold tracking-tight text-white sm:text-4xl animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
          >
            {t.citationTitle}
          </h2>
          <p
            className={`mt-3 text-sm text-white/35 animate-initial ${
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
            <button onClick={handleCopy} className={`copy-btn ${copied ? 'copied' : ''}`}>
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
          <div className="code-content font-mono text-[13px] leading-relaxed">
            <div style={{ color: '#c792ea' }}>@article<span style={{ color: '#e8ecf4' }}>{'{'}</span><span style={{ color: '#c3e88d' }}>mandol2026</span>,</div>
            <div className="pl-4"><span style={{ color: '#82aaff' }}>title</span>   = {'{Mandol: An In-Memory Layered Memory System'}</div>
            <div className="pl-8">{'for Long-Term Conversational Agents},'}</div>
            <div className="pl-4"><span style={{ color: '#82aaff' }}>author</span>  = {'{Yuhan Zhang and Zhiyuan Guo and Ziheng Zeng'}</div>
            <div className="pl-8">{'and Wei Wang and Wentao Wu and Lijie Xu},'}</div>
            <div className="pl-4"><span style={{ color: '#82aaff' }}>journal</span> = {'{arXiv preprint arXiv:260x.xxxxx},'}</div>
            <div className="pl-4"><span style={{ color: '#82aaff' }}>year</span>    = {'{2026}'}</div>
            <div>{'}'}</div>
          </div>
        </div>

        <p
          className={`mt-5 text-center text-[12px] text-white/20 animate-initial ${
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
