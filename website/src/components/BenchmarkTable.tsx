import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/translations';

// ── LoCoMo table data ──────────────────────────────────────────────
const locomoBackbones = ['GPT-4o-mini', 'GPT-4.1-mini'] as const;

interface LocomoSystem {
  name: string;
  data: string[];
}

const locomoSystems: Record<string, LocomoSystem[]> = {
  'GPT-4o-mini': [
    { name: 'Mem0',    data: ['1.0k', '66.71', '58.16', '55.45', '40.62', '61.00'] },
    { name: 'MemU',    data: ['4.0k', '72.77', '62.41', '33.96', '46.88', '61.15'] },
    { name: 'MemOS',   data: ['2.5k', '81.45', '69.15', '72.27', '60.42', '75.87'] },
    { name: 'Zep',     data: ['1.4k', '88.11', '71.99', '74.45', '66.67', '81.06'] },
    { name: 'EverMemOS', data: ['2.5k', '91.68', '82.74', '79.34', '70.14', '86.13'] },
    { name: 'Mandol',  data: ['2.0k', '93.82', '85.11', '89.10', '65.63', '89.48'] },
  ],
  'GPT-4.1-mini': [
    { name: 'Mem0',    data: ['1.0k', '68.97', '61.70', '58.26', '50.00', '64.20'] },
    { name: 'MemU',    data: ['4.0k', '74.91', '72.34', '43.61', '54.17', '66.67'] },
    { name: 'MemOS',   data: ['2.5k', '85.37', '79.43', '75.08', '64.58', '80.76'] },
    { name: 'Zep',     data: ['1.4k', '90.84', '81.91', '77.26', '75.00', '85.22'] },
    { name: 'EverMemOS', data: ['2.3k', '95.32', '89.01', '90.13', '77.43', '91.97'] },
    { name: 'Mandol',  data: ['1.9k', '95.36', '92.20', '87.85', '79.17', '92.21'] },
  ],
};

const locomoCols = ['Avg. Tok', 'Single', 'Multi', 'Temp.', 'Open', 'Overall'];

// ── LongMemEval table data ─────────────────────────────────────────
const longmemBackbones = ['GPT-4o-mini', 'GPT-4.1-mini'] as const;

interface LongmemSystem {
  name: string;
  data: string[];
}

const longmemSystems: Record<string, LongmemSystem[]> = {
  'GPT-4o-mini': [
    { name: 'MemU',   data: ['0.5k', '76.70', '19.60', '17.30', '42.10', '41.00', '67.10', '38.40'] },
    { name: 'Mem0',   data: ['1.1k', '90.00', '26.78', '72.18', '63.15', '66.67', '82.86', '66.40'] },
    { name: 'Zep',    data: ['1.6k', '53.30', '75.00', '54.10', '47.40', '74.40', '92.90', '63.80'] },
    { name: 'MemOS',  data: ['1.4k', '96.67', '67.86', '77.44', '70.67', '74.26', '95.71', '77.80'] },
    { name: 'Mandol', data: ['2.1k', '96.67', '98.21', '78.95', '74.44', '88.46', '97.14', '85.00'] },
  ],
  'GPT-4.1-mini': [
    { name: 'EverMemOS', data: ['2.8k', '93.33', '85.71', '77.44', '73.68', '89.74', '97.14', '83.00'] },
    { name: 'Mandol',    data: ['2.3k', '96.67', '98.21', '87.22', '77.44', '89.74', '98.57', '88.40'] },
  ],
};

const longmemCols = ['Avg. Tok', 'SS-Pref', 'SS-Asst', 'Temporal', 'Multi-S', 'Know. Upd.', 'SS-User', 'Overall'];

// ── Latency data (milliseconds) ───────────────────────────────────

interface LatencySystem {
  name: string;
  search: [string, string, string];
  add: [string, string, string];
  reproduced?: boolean;
  isMandol?: boolean;
}

const serverLatencySystems: LatencySystem[] = [
  { name: 'MemU', search: ['63000.7', '60539.5', '47554.5'], add: ['12070.6', '7273.1', '5077.9'] },
  {
    name: 'EverMemOS',
    search: ['37192.4', '35220.4', '20092.1'],
    add: ['790.2', '555.5', '317.7'],
    reproduced: true,
  },
  { name: 'Mem0', search: ['4637.0', '1397.0', '1089.0'], add: ['2841.0', '1650.0', '888.0'] },
  { name: 'Zep', search: ['5348.7', '614.8', '571.7'], add: ['375.1', '254.5', '239.0'] },
  { name: 'MemOS', search: ['777.1', '528.4', '440.5'], add: ['376.4', '211.6', '191.9'] },
  {
    name: 'Mandol',
    search: ['94.8', '88.5', '82.2'],
    add: ['67.3', '46.9', '39.7'],
    isMandol: true,
  },
];

const localLatency = {
  search: ['211.6', '186.8', '166.5'],
  add: ['51.6', '42.1', '37.4'],
} as const;

// ── Reusable table renderer ────────────────────────────────────────

function DataTable({
  backbone,
  cols,
  systems,
  minWidth,
}: {
  backbone: string;
  cols: string[];
  systems: { name: string; data: string[] }[];
  minWidth: string;
}) {
  return (
    <div className="mb-8 flex justify-center overflow-x-auto">
      <table className="bench-table" style={{ minWidth }}>
        <thead>
          <tr>
            <th>
              <span className="site-text-muted text-xs font-medium">{backbone}</span>
            </th>
            {cols.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {systems.map((sys) => (
            <tr key={sys.name} className={sys.name === 'Mandol' ? 'mandol-row' : ''}>
              <td className="sys-name">{sys.name}</td>
              {sys.data.map((v, i) => (
                <td key={i} className="tabular-nums">
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────

export default function BenchmarkTable(): React.JSX.Element {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];
  const isZh = locale === 'zh-Hans';
  const mandolServerLatency = serverLatencySystems[serverLatencySystems.length - 1];

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.05 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={ref} className="section-darker py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        {/* Section header */}
        <div className="mb-16 text-center">
          <h2
            className={`site-heading text-3xl font-bold tracking-tight sm:text-4xl lg:text-5xl animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
          >
            {isZh ? (
              <>
                基准<span className="gradient-text-blue">性能对比</span>
              </>
            ) : (
              <>
                Benchmark <span className="gradient-text-blue">Performance</span>
              </>
            )}
          </h2>
          <p
            className={`site-text-subtle mt-3 text-sm animate-initial ${
              visible ? 'animate-fade-in-up' : ''
            }`}
            style={{ animationDelay: '0.1s' }}
          >
            {t.benchmarkSubtitle}
          </p>
        </div>

        {/* ─── LoCoMo Table ─── */}
        <div
          className={`animate-initial mb-16 ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.15s' }}
        >
          <h3 className="site-text mb-5 text-center text-lg font-semibold">
            {t.benchmarkLocomoTitle}
          </h3>

          {locomoBackbones.map((backbone) => (
            <DataTable
              key={backbone}
              backbone={backbone}
              cols={locomoCols}
              systems={locomoSystems[backbone]}
              minWidth="720px"
            />
          ))}

          <p className="site-text-faint mt-4 text-center text-[12px] leading-relaxed max-w-3xl mx-auto">
            {t.benchmarkLocomoNote}
          </p>
        </div>

        {/* ─── LongMemEval Table ─── */}
        <div
          className={`animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.25s' }}
        >
          <h3 className="site-text mb-5 text-center text-lg font-semibold">
            {t.benchmarkLongmemTitle}
          </h3>

          {longmemBackbones.map((backbone) => (
            <DataTable
              key={backbone}
              backbone={backbone}
              cols={longmemCols}
              systems={longmemSystems[backbone]}
              minWidth="840px"
            />
          ))}

          <p className="site-text-faint mt-4 text-center text-[12px] leading-relaxed max-w-3xl mx-auto">
            {t.benchmarkLongmemNote}
          </p>
        </div>

        {/* ─── Latency Performance ─── */}
        <div
          className={`latency-performance animate-initial mt-20 pt-16 ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.35s' }}
        >
          <div className="mb-8 text-center">
            <h3 className="site-heading text-2xl font-bold sm:text-3xl">{t.benchmarkLatencyTitle}</h3>
            <p className="site-text-muted mx-auto mt-3 max-w-2xl text-sm leading-relaxed">
              {t.benchmarkLatencySubtitle}
            </p>
          </div>

          <div className="latency-metric-grid" aria-label={t.benchmarkLatencyTitle}>
            {[
              [`${mandolServerLatency.search[2]} ms`, t.benchmarkMetricSearch, t.benchmarkMetricServerContext],
              [`${mandolServerLatency.add[2]} ms`, t.benchmarkMetricAdd, t.benchmarkMetricServerContext],
              ['5.4×', t.benchmarkMetricRetrievalSpeedup, t.benchmarkMetricBaselineContext],
              ['4.8×', t.benchmarkMetricInsertionSpeedup, t.benchmarkMetricBaselineContext],
            ].map(([value, label, context]) => (
              <article className="latency-metric" key={label}>
                <p className="latency-metric-value tabular-nums">{value}</p>
                <h4 className="latency-metric-label">{label}</h4>
                <p className="latency-metric-context">{context}</p>
              </article>
            ))}
          </div>

          <div className="latency-server-header">
            <div>
              <h4 className="site-heading text-lg font-semibold">{t.benchmarkServerTitle}</h4>
              <p className="site-text-muted mt-1 text-sm">{t.benchmarkServerEnvironment}</p>
            </div>
            <span className="latency-unit">ms</span>
          </div>

          <div
            className="latency-table-scroll"
            role="region"
            aria-label={`${t.benchmarkServerTitle}: ${t.benchmarkLatencyTitle}`}
            tabIndex={0}
          >
            <table className="bench-table latency-table">
              <caption className="sr-only">{t.benchmarkServerEnvironment}</caption>
              <thead>
                <tr>
                  <th rowSpan={2} scope="col">{t.benchmarkSystem}</th>
                  <th colSpan={3} scope="colgroup">{t.benchmarkSearch}</th>
                  <th colSpan={3} scope="colgroup">{t.benchmarkAdd}</th>
                </tr>
                <tr>
                  <th scope="col">P99</th>
                  <th scope="col">P90</th>
                  <th scope="col">{t.benchmarkMean}</th>
                  <th scope="col">P99</th>
                  <th scope="col">P90</th>
                  <th scope="col">{t.benchmarkMean}</th>
                </tr>
              </thead>
              <tbody>
                {serverLatencySystems.map((system) => (
                  <tr key={system.name} className={system.isMandol ? 'mandol-row' : ''}>
                    <th scope="row" className="sys-name">
                      {system.name}
                      {system.reproduced ? <sup>†</sup> : null}
                      {system.isMandol ? <span className="latency-ours"> ({t.benchmarkOurs})</span> : null}
                    </th>
                    {[...system.search, ...system.add].map((value, index) => (
                      <td key={index} className="tabular-nums">{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <aside className="latency-local" aria-labelledby="local-latency-title">
            <div className="latency-local-copy">
              <p className="latency-local-eyebrow">{t.benchmarkLocalEyebrow}</p>
              <h4 id="local-latency-title" className="site-heading text-lg font-semibold">
                {t.benchmarkLocalTitle}
              </h4>
              <p className="site-text mt-2 text-sm leading-relaxed">{t.benchmarkLocalDescription}</p>
              <p className="site-text-muted mt-2 text-xs leading-relaxed">{t.benchmarkLocalSeparate}</p>
            </div>
            <dl className="latency-local-values">
              <div>
                <dt>{t.benchmarkSearch} · 5 QPS</dt>
                <dd className="tabular-nums">
                  P99 {localLatency.search[0]} · P90 {localLatency.search[1]} · {t.benchmarkMean}{' '}
                  {localLatency.search[2]}
                </dd>
              </div>
              <div>
                <dt>{t.benchmarkAdd} · 10 QPS</dt>
                <dd className="tabular-nums">
                  P99 {localLatency.add[0]} · P90 {localLatency.add[1]} · {t.benchmarkMean} {localLatency.add[2]}
                </dd>
              </div>
            </dl>
          </aside>

          <div className="latency-provenance">
            <p>{t.benchmarkLatencyConditions}</p>
            <p>{t.benchmarkLatencyEvermemNote}</p>
            <p>
              {t.benchmarkLatencySourcePrefix}{' '}
              <a href="https://github.com/AgentCombo/Mandol/tree/paper-repro">
                {t.benchmarkLatencySourceLink}
              </a>
              {t.benchmarkLatencySourceSuffix}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
