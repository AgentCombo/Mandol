import React, { useEffect, useRef, useState } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/data/translations';

const highlights = [
  { value: '92.21%', labelKey: 'highlightLoCoMo' as const, color: '#60a5fa' },
  { value: '88.40%', labelKey: 'highlightLongMemEval' as const, color: '#06b6d4' },
  { value: '5.4×', labelKey: 'highlightRetrievalSpeedup' as const, color: '#a78bfa' },
  { value: '4.8×', labelKey: 'highlightInsertionSpeedup' as const, color: '#f59e0b' },
];

export default function HeroSection(): JSX.Element {
  const [visible, setVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const handleCopyInstall = () => {
    navigator.clipboard.writeText('pip install mandol');
  };

  return (
    <section
      ref={sectionRef}
      className="section-hero relative overflow-hidden pt-24 pb-20 sm:pt-32 sm:pb-28"
    >
      {/* Background glow orbs */}
      <div
        className="pointer-events-none absolute -top-32 left-1/4 h-[500px] w-[600px] rounded-full opacity-20"
        style={{
          background: 'radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%)',
          filter: 'blur(80px)',
          animation: visible ? 'float 8s ease-in-out infinite' : 'none',
        }}
      />
      <div
        className="pointer-events-none absolute -bottom-40 right-1/4 h-[400px] w-[500px] rounded-full opacity-15"
        style={{
          background: 'radial-gradient(circle, rgba(6,182,212,0.2) 0%, transparent 70%)',
          filter: 'blur(60px)',
          animation: visible ? 'float 10s ease-in-out infinite 1s' : 'none',
        }}
      />

      {/* Subtle grid pattern */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      <div className="relative z-10 mx-auto max-w-6xl px-6">
        {/* Badge row */}
        <div
          className={`badge-row mb-10 animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.1s' }}
        >
          <a
            href="https://github.com/AgentCombo/Mandol"
            target="_blank"
            rel="noopener noreferrer"
            className="badge-item"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            GitHub
          </a>
          <a
            href="https://arxiv.org/abs/260x.xxxxx"
            target="_blank"
            rel="noopener noreferrer"
            className="badge-item"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            Paper
          </a>
          <a href="/Mandol/docs/" className="badge-item">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
            </svg>
            {t.heroBadgeDocs}
          </a>
        </div>

        {/* Main title */}
        <h1
          className={`text-center text-6xl font-extrabold tracking-tight sm:text-7xl lg:text-8xl animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.25s' }}
        >
          <span className="gradient-text">Mandol</span>
        </h1>

        {/* Subtitle */}
        <p
          className={`mx-auto mt-6 max-w-2xl text-center text-lg text-white/60 sm:text-xl animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.4s' }}
        >
          {t.heroTitle_sub1}
          <br />
          <span className="text-white/35">{t.heroTitle_sub2}</span>
        </p>

        {/* Description */}
        <p
          className={`mx-auto mt-5 max-w-xl text-center text-sm leading-relaxed text-white/40 animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.5s' }}
        >
          {t.heroDescription}
        </p>

        {/* CTA buttons */}
        <div
          className={`mt-10 flex flex-wrap items-center justify-center gap-3 animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.6s' }}
        >
          <button onClick={handleCopyInstall} className="btn-ghost group">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="opacity-50">
              <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
              <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
              <line x1="6" y1="6" x2="6.01" y2="6" />
              <line x1="6" y1="18" x2="6.01" y2="18" />
            </svg>
            <span className="font-mono text-sm">{t.heroPipInstall}</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="opacity-40 group-hover:opacity-80">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
          </button>
          <a href="/Mandol/docs/" className="btn-primary">
            {t.heroGetStarted}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
          </a>
          <a href="https://github.com/AgentCombo/Mandol" target="_blank" rel="noopener noreferrer" className="btn-secondary">
            <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            {t.heroViewOnGitHub}
          </a>
        </div>

        {/* Highlights row */}
        <div
          className={`mt-20 flex flex-wrap items-center justify-center gap-x-10 gap-y-5 sm:gap-x-14 animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.7s' }}
        >
          {highlights.map((h, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-2xl sm:text-3xl font-extrabold tracking-tight tabular-nums" style={{ color: h.color }}>
                {h.value}
              </span>
              <span className="text-xs sm:text-sm text-white/35 font-medium whitespace-nowrap leading-tight">
                {t[h.labelKey]}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
