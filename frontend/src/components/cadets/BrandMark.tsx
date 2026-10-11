export function BrandMark({ size = 36, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" className={className} aria-hidden="true">
      <path d="M5 9h22M16 9v30" stroke="currentColor" strokeWidth="4" strokeLinecap="square" />
      <path d="M28 9h4c8 0 12 6 12 15s-4 15-12 15h-4V22h9" stroke="currentColor" strokeWidth="4" strokeLinejoin="round" />
      <path d="M5 39h11" stroke="var(--app-accent)" strokeWidth="4" />
    </svg>
  );
}
