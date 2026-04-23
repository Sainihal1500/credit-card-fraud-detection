
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { CoolingData } from '../DeviceSimulator';

interface LiveChartProps {
  data: CoolingData[];
}

export function LiveChart({ data }: LiveChartProps) {
  return (
    <div className="glass-card" style={{ height: '400px', padding: '1.5rem', paddingBottom: '3rem' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Temperature Differential Analysis</h3>
        <p style={{ margin: '0.5rem 0 0 0', color: 'var(--text-dim)', fontSize: '0.875rem' }}>
          Base panel expected temp vs actual cooled temperature
        </p>
      </div>
      
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis 
            dataKey="time" 
            stroke="var(--text-dim)" 
            tick={{ fill: 'var(--text-dim)', fontSize: 12 }} 
            minTickGap={30}
          />
          <YAxis 
            stroke="var(--text-dim)" 
            tick={{ fill: 'var(--text-dim)', fontSize: 12 }}
            domain={['dataMin - 5', 'dataMax + 5']}
            unit="°C"
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'rgba(30, 41, 59, 0.9)', backdropFilter: 'blur(10px)', border: '1px solid var(--border-glass)', borderRadius: '0.5rem', color: '#fff' }}
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          <Line 
            type="monotone" 
            dataKey="basePanelTemp" 
            name="Uncooled Panel" 
            stroke="var(--neon-red)" 
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="cooledPanelTemp" 
            name="Radiatively Cooled Panel" 
            stroke="var(--neon-blue)" 
            strokeWidth={3}
            dot={false}
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="ambientTemp" 
            name="Ambient Air" 
            stroke="var(--neon-green)" 
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
