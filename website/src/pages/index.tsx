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
import { translations, type Locale } from '@site/src/translations';

function HomeFooter(): ReactNode {
  const { i18n } = useDocusaurusContext();
  const locale = (i18n.currentLocale || 'en') as Locale;
  const t = translations[locale];

  return (
    <footer className="section-darker site-footer py-12">
      <div className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
          <div className="site-text-faint text-[13px]">
            &copy; {new Date().getFullYear()} {t.footerCopyright}
          </div>
          <div className="flex gap-6 text-[13px]">
            <a
              href="https://agentcombo.github.io/Mandol/docs/"
              className="site-text-muted transition-colors no-underline hover:text-primary-500"
            >
              {t.footerDocs}
            </a>
            <a
              href="https://github.com/AgentCombo/Mandol"
              target="_blank"
              rel="noopener noreferrer"
              className="site-text-muted transition-colors no-underline hover:text-primary-500"
            >
              {t.footerGitHub}
            </a>
            <a
              href="https://github.com/AgentCombo/Mandol/discussions"
              target="_blank"
              rel="noopener noreferrer"
              className="site-text-muted transition-colors no-underline hover:text-primary-500"
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
          ? 'Mandol — 面向智能体检索系统的内存语义记忆运行时'
          : 'Mandol — An In-Memory Semantic Memory Runtime for Agent Retrieval Systems'
      }
      description={
        locale === 'zh-Hans'
          ? 'Mandol 当前公开 MemoryUnit、SemanticMap、SemanticGraph、MultiRetriever 与三塔检索组件。'
          : 'Mandol currently exposes MemoryUnit, SemanticMap, SemanticGraph, MultiRetriever, and three-tower retrieval components.'
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
