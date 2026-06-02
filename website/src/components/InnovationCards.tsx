import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/data/translations';

interface Innovation {
  icon: string;
  title: string;
  tag: string;
  tagColor: string;
  description: string;
  points: string[];
}

function buildInnovations(t: typeof translations['en']): Innovation[] {
  return [
    {
      icon: '🏗️',
      title: t.innovation1_title,
      tag: 'Theoretical',
      tagColor: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
      description: t.innovation1_desc,
      points: t.innovation1_points,
    },
    {
      icon: '💾',
      title: t.innovation2_title,
      tag: 'Architecture',
      tagColor: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
      description: t.innovation2_desc,
      points: t.innovation2_points,
    },
    {
      icon: '🔍',
      title: t.innovation3_title,
      tag: 'Retrieval',
      tagColor: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
      description: t.innovation3_desc,
      points: t.innovation3_points,
    },
  ];
}

function InnovationCard({
  inv,
  visible,
  delay,
}: {
  inv: Innovation;
  visible: boolean;
  delay: number;
}) {
  return (
    <div
      className={`card-glow flex flex-col p-7 animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
      style={{ animationDelay: `${delay}s` }}
    >
      <div className="mb-4 text-3xl">{inv.icon}</div>
      <div className="mb-2 flex items-center gap-3">
        <h3 className="text-lg font-semibold text-white/90">{inv.title}</h3>
      </div>
      <span
        className={`mb-4 inline-block self-start rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${inv.tagColor}`}
      >
        {inv.tag}
      </span>
      <p className="mb-5 text-sm leading-relaxed text-white/50">{inv.description}</p>
      <ul className="mt-auto space-y-2.5">
        {inv.points.map((p, i) => (
          <li key={i} className="flex items-start gap-2.5 text-[13px] leading-relaxed text-white/45">
            <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary-400/50" />
            {p}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function InnovationCards(): JSX.Element {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];
  const innovations = buildInnovations(t);

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

  return (
    <section ref={ref} className="section-dark py-24 sm:py-32">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-16 text-center">
          <h2
            className={`text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
          >
            {locale === 'zh-Hans' ? (
              <>
                核心<span className="gradient-text-blue">创新</span>
              </>
            ) : (
              <>
                Core <span className="gradient-text-blue">Innovations</span>
              </>
            )}
          </h2>
          <p
            className={`mt-3 text-sm text-white/35 animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
            style={{ animationDelay: '0.1s' }}
          >
            {t.innovationsSubtitle}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {innovations.map((inv, i) => (
            <InnovationCard
              key={inv.title}
              inv={inv}
              visible={visible}
              delay={0.15 + i * 0.12}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
