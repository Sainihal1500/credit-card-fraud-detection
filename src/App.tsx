import { useState } from 'react';
import { useDeviceSimulator } from './DeviceSimulator';
import { MetricsCard } from './components/MetricsCard';
import { LiveChart } from './components/LiveChart';
import { 
  Activity, 
  Thermometer, 
  ThermometerSnowflake, 
  ThermometerSun, 
  Zap, 
  BatteryCharging 
} from 'lucide-react';
import './index.css';

function App() {
  const { timeSeriesData, currentMetrics } = useDeviceSimulator();
  const [deviceConnected, setDeviceConnected] = useState(true);

  if (!currentMetrics) return <div style={{ padding: '2rem', color: '#fff' }}>Initializing Hardware Interface...</div>;

  const coolingDrop = currentMetrics.basePanelTemp - currentMetrics.cooledPanelTemp;

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
      
      <header className="header">
        <div>
          <h1 style={{ margin: 0, fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.025em', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Zap className="glow-blue" style={{ color: 'var(--neon-blue)' }} size={32} />
            AeroCool Nexus <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}>| Real-Time Array Monitor</span>
          </h1>
        </div>
        
        <div className="status-badge" onClick={() => setDeviceConnected(!deviceConnected)} style={{ cursor: 'pointer', background: deviceConnected ? 'rgba(74, 222, 128, 0.1)' : 'rgba(248, 113, 113, 0.1)', color: deviceConnected ? 'var(--neon-green)' : 'var(--neon-red)', borderColor: deviceConnected ? 'rgba(74, 222, 128, 0.2)' : 'rgba(248, 113, 113, 0.2)' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: deviceConnected ? 'var(--neon-green)' : 'var(--neon-red)' }} className={deviceConnected ? "animate-pulse-slow" : ""} />
          {deviceConnected ? 'HARDWARE SYNC: LIVE' : 'HARDWARE DISCONNECTED'}
        </div>
      </header>

      {deviceConnected ? (
        <>
          <div className="dashboard-grid">
            <MetricsCard 
              title="Temperature Reduction (∆T)"
              value={coolingDrop.toFixed(1)}
              unit="°C"
              icon={<ThermometerSnowflake size={24} color="var(--neon-blue)" />}
              subtitle="Current passive radiative cooling delta"
              glowColor="blue"
              valueColor="var(--neon-blue)"
            />
            
            <MetricsCard 
              title="Instantaneous Power Gain"
              value={`+${currentMetrics.powerOutputGain.toFixed(1)}`}
              unit="W"
              icon={<Activity size={24} color="var(--neon-green)" />}
              subtitle="Equivalent power boost from thermal relief (400W base panel)"
              glowColor="green"
              valueColor="var(--neon-green)"
            />
            
            <MetricsCard 
              title="Cumulative Energy Saved"
              value={currentMetrics.energySaved.toFixed(2)}
              unit="Wh"
              icon={<BatteryCharging size={24} color="var(--neon-green)" />}
              subtitle="Extra energy generated this session"
            />
            
            <MetricsCard 
              title="Current Panel Interface Temp"
              value={currentMetrics.cooledPanelTemp.toFixed(1)}
              unit="°C"
              icon={<Thermometer size={24} color="var(--text-main)" />}
              subtitle={`Ambient: ${currentMetrics.ambientTemp.toFixed(1)}°C | Uncooled Est: ${currentMetrics.basePanelTemp.toFixed(1)}°C`}
            />
          </div>

          <div style={{ marginTop: '2rem' }}>
            <LiveChart data={timeSeriesData} />
          </div>
        </>
      ) : (
        <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem', border: '1px solid var(--neon-red)' }}>
          <ThermometerSun size={48} color="var(--neon-red)" style={{ margin: '0 auto 1rem auto', opacity: 0.8 }} />
          <h2 style={{ margin: '0 0 1rem 0', color: 'var(--neon-red)' }}>Connection Lost</h2>
          <p style={{ color: 'var(--text-dim)', maxWidth: '500px', margin: '0 auto' }}>
            Could not retrieve data from the field apparatus. Please check the network bridge or hardware power supply. Click the status badge to attempt reconnection.
          </p>
        </div>
      )}
      
    </div>
  );
}

export default App;
