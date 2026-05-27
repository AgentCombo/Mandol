import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import FeaturesGrid from '@site/src/components/FeaturesGrid';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="section-dark py-32">
      <div className="container mx-auto px-6 text-center">
        <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl">
          <span className="gradient-text">{siteConfig.title}</span>
        </h1>
        <p className="mt-6 text-xl text-white/60 max-w-2xl mx-auto">
          {siteConfig.tagline}
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            className="rounded-lg bg-primary-500 px-6 py-3 font-semibold text-white no-underline hover:bg-primary-600 transition-colors"
            to="/docs/getting-started/installation">
            Get Started
          </Link>
          <Link
            className="rounded-lg border border-white/20 px-6 py-3 font-semibold text-white/80 no-underline hover:border-white/40 transition-colors"
            to="https://github.com/AgentCombo/Mandol">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="An In-Memory Agent Memory System for Long-Term Conversations">
      <HomepageHeader />
      <main>
        <FeaturesGrid />
      </main>
    </Layout>
  );
}
