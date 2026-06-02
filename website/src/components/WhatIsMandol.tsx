import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/data/translations';

export default function WhatIsMandol(): JSX.Element {
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

  return (
    <section ref={ref} className="section-darker py-24 sm:py-32">
      <div className="mx-auto max-w-4xl px-6">
        {/* Section header */}
        <div className="mb-16 text-center">
          <h2
            className={`text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
          >
            {locale === 'zh-Hans' ? (
              <>
                什么是 <span className="gradient-text-blue">Mandol</span>？
              </>
            ) : (
              <>
                What is <span className="gradient-text-blue">Mandol</span>?
              </>
            )}
          </h2>
        </div>

        {/* Core description */}
        <div
          className={`mx-auto max-w-2xl text-center animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.15s' }}
        >
          <p className="text-[15px] leading-relaxed text-white/55">{t.whatIsDesc1}</p>
          <p className="mt-4 text-[15px] leading-relaxed text-white/45">{t.whatIsDesc2}</p>
        </div>

        {/* Architecture overview image */}
        <div
          className={`mt-14 flex justify-center animate-initial ${
            visible ? 'animate-fade-in-up' : ''
          }`}
          style={{ animationDelay: '0.3s' }}
        >
          <img
            src="/Mandol/img/mandol-overview.png"
            alt="Mandol System Architecture Overview"
            className="w-full max-w-3xl rounded-2xl"
            style={{
              boxShadow: '0 4px 60px rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
            }}
          />
        </div>
      </div>
    </section>
  );
}
