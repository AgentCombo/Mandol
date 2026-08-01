import React, { type ReactNode } from 'react';
import clsx from 'clsx';
import useIsBrowser from '@docusaurus/useIsBrowser';
import { useColorMode, useThemeConfig } from '@docusaurus/theme-common';
import IconDarkMode from '@theme/Icon/DarkMode';
import IconLightMode from '@theme/Icon/LightMode';
import type { Props } from '@theme/Navbar/ColorModeToggle';

import styles from './styles.module.css';

export default function NavbarColorModeToggle({ className }: Props): ReactNode {
  const isBrowser = useIsBrowser();
  const { disableSwitch } = useThemeConfig().colorMode;
  const { colorMode, setColorMode } = useColorMode();

  if (disableSwitch) return null;

  const nextMode = colorMode === 'dark' ? 'light' : 'dark';

  return (
    <div className={clsx(styles.toggle, className)}>
      <button
        type="button"
        className={clsx('clean-btn', styles.toggleButton)}
        disabled={!isBrowser}
        title={`Switch to ${nextMode} mode`}
        aria-label={`Switch to ${nextMode} mode`}
        onClick={() => setColorMode(nextMode)}
      >
        {nextMode === 'light' ? (
          <IconLightMode aria-hidden />
        ) : (
          <IconDarkMode aria-hidden />
        )}
      </button>
    </div>
  );
}
