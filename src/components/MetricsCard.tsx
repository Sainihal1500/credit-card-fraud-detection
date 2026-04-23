import type { ReactNode } from 'react';

interface MetricsCardProps {
  title: string;
  value: string | number;
  unit: string;
  icon: ReactNode;
  subtitle?: string;
  glowColor?: 'blue' | 'green' | 'red';
  valueColor?: string;
}

export function MetricsCard({ title, value, unit, icon, subtitle, glowColor, valueColor }: MetricsCardProps) {
  const glowClass = glowColor ? `glow-${glowColor}` : '';
  
  return (
    <div className="glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-dim)', fontWeight: 500 }}>{title}</h3>
        <div style={{ color: 'var(--text-dim)' }}>{icon}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.25rem' }}>
        <span className={glowClass} style={{ fontSize: '2.5rem', fontWeight: 700, color: valueColor || 'var(--text-main)', letterSpacing: '-0.025em' }}>
          {value}
        </span>
        <span style={{ fontSize: '1rem', color: 'var(--text-dim)', fontWeight: 500 }}>{unit}</span>
      </div>
      {subtitle && (
        <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.875rem', color: 'var(--text-dim)' }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}
