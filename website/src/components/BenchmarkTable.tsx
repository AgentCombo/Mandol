import React, { useEffect, useRef, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { translations, type Locale } from '@site/src/data/translations';

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
              <span className="text-white/50 text-xs font-medium">{backbone}</span>
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

export default function BenchmarkTable(): JSX.Element {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];
  const isZh = locale === 'zh-Hans';

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
            className={`text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl animate-initial ${
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
            className={`mt-3 text-sm text-white/35 animate-initial ${
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
          <h3 className="mb-5 text-center text-lg font-semibold text-white/65">
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

          <p className="mt-4 text-center text-[12px] leading-relaxed text-white/20 max-w-3xl mx-auto">
            {t.benchmarkLocomoNote}
          </p>
        </div>

        {/* ─── LongMemEval Table ─── */}
        <div
          className={`animate-initial ${visible ? 'animate-fade-in-up' : ''}`}
          style={{ animationDelay: '0.25s' }}
        >
          <h3 className="mb-5 text-center text-lg font-semibold text-white/65">
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

          <p className="mt-4 text-center text-[12px] leading-relaxed text-white/20 max-w-3xl mx-auto">
            {t.benchmarkLongmemNote}
          </p>
        </div>
      </div>
    </section>
  );
}
