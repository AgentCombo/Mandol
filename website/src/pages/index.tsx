import type { ReactNode } from 'react';
import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HeroSection from '@site/src/components/HeroSection';
import WhatIsMandol from '@site/src/components/WhatIsMandol';
import InnovationCards from '@site/src/components/InnovationCards';
import BenchmarkTable from '@site/src/components/BenchmarkTable';
import QuickStartTabs from '@site/src/components/QuickStartTabs';
import CitationBlock from '@site/src/components/CitationBlock';
import { translations, type Locale } from '@site/src/data/translations';

function HomeFooter(): ReactNode {
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];

  return (
    <footer className="section-darker border-t border-white/[0.04] py-12">
      <div className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
          <div className="text-[13px] text-white/25">
            &copy; {new Date().getFullYear()} {t.footerCopyright}
          </div>
          <div className="flex gap-6 text-[13px]">
            <a
              href="/Mandol/docs/"
              className="text-white/30 hover:text-white/60 transition-colors no-underline"
            >
              {t.footerDocs}
            </a>
            <a
              href="https://github.com/AgentCombo/Mandol"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white/30 hover:text-white/60 transition-colors no-underline"
            >
              {t.footerGitHub}
            </a>
            <a
              href="https://github.com/AgentCombo/Mandol/discussions"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white/30 hover:text-white/60 transition-colors no-underline"
            >
              {t.footerCommunity}
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function Home(): ReactNode {
  const { i18n } = useDocusaurusContext();
  const locale = i18n.currentLocale || 'en';

  return (
    <Layout
      title={
        locale === 'zh-Hans'
          ? 'Mandol — 面向长期对话Agent的内存级分层记忆系统'
          : 'Mandol — An In-Memory Layered Memory System for Long-Term Conversational Agents'
      }
      description={
        locale === 'zh-Hans'
          ? 'Mandol 是一个面向长期对话Agent的内存级分层记忆系统。LoCoMo 92.21% | LongMemEval 88.40% | 5.8× 构建加速 | 5.4× 检索加速。'
          : 'Mandol is an in-memory, layered memory system for long-term conversational agents. LoCoMo 92.21% | LongMemEval 88.40% | 5.8× build speed | 5.4× retrieval speed.'
      }
    >
      <main className="homepage-layout">
        <HeroSection />
        <hr className="section-divider" />
        <WhatIsMandol />
        <hr className="section-divider" />
        <InnovationCards />
        <hr className="section-divider" />
        <BenchmarkTable />
        <hr className="section-divider" />
        <QuickStartTabs />
        <hr className="section-divider" />
        <CitationBlock />
        <HomeFooter />
      </main>
    </Layout>
  );
}
