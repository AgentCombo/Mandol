export async function copyText(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Fall through to the selection-based copy path.
  }

  const textarea = document.createElement('textarea');
  textarea.value = value;
  textarea.readOnly = true;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);

  try {
    textarea.select();
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    textarea.remove();
  }
}
