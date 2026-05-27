import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/installation'],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'core-concepts/memory-space',
        'core-concepts/memory-unit',
        'core-concepts/semantic-graph',
        'core-concepts/semantic-map',
        'core-concepts/entities-events',
        'core-concepts/sessions',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/hexagonal',
        'architecture/ports-adapters',
        'architecture/adding-backends',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/env-vars',
        'configuration/yaml-config',
      ],
    },
    'api',
  ],
};

export default sidebars;
