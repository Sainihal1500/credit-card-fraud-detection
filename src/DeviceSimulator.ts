import { useState, useEffect } from 'react';

export interface CoolingData {
  time: string;
  ambientTemp: number;     // °C
  basePanelTemp: number;   // °C
  cooledPanelTemp: number; // °C
  powerOutputGain: number; // Watts
  energySaved: number;     // Wh cumulative
}

export function useDeviceSimulator() {
  const [data, setData] = useState<CoolingData[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<CoolingData | null>(null);

  useEffect(() => {
    // Initial data generation
    let energyAccumulator = 0;
    const initialData: CoolingData[] = [];
    
    // Generate last 20 data points
    for (let i = 20; i >= 0; i--) {
      const now = new Date();
      now.setSeconds(now.getSeconds() - i * 5); // 5 sec intervals
      
      const timeStr = now.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' });
      
      // Simulate realism
      const baseVariation = Math.sin(i / 5) * 5; 
      const ambient = 30 + baseVariation + Math.random();
      const basePanel = ambient + 25 + Math.random() * 2; // Panels get hot!
      
      // Radiative cooling reduces panel temp significantly
      const coolingEffect = 12 + Math.random() * 3; 
      const cooledPanel = basePanel - coolingEffect;
      
      // Every 1C above 25C reduces output by 0.4%
      // So lowering temp increases power. Assuming a 400W panel base.
      const outputGain = (coolingEffect * 0.004) * 400; 
      
      energyAccumulator += (outputGain * (5 / 3600)); // power * hours = energy

      initialData.push({
        time: timeStr,
        ambientTemp: Number(ambient.toFixed(1)),
        basePanelTemp: Number(basePanel.toFixed(1)),
        cooledPanelTemp: Number(cooledPanel.toFixed(1)),
        powerOutputGain: Number(outputGain.toFixed(2)),
        energySaved: Number(energyAccumulator.toFixed(3))
      });
    }

    setData(initialData);
    setCurrentMetrics(initialData[initialData.length - 1]);

    // Setup interval for live updates
    const interval = setInterval(() => {
      setData(prevData => {
        const last = prevData[prevData.length - 1];
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' });
        
        // Random walk for data
        const ambient = last.ambientTemp + (Math.random() - 0.5) * 0.5;
        const basePanel = ambient + 25 + (Math.random() - 0.5);
        const coolingEffect = 12 + (Math.random() - 0.5) * 2;
        const cooledPanel = basePanel - coolingEffect;
        
        const outputGain = (coolingEffect * 0.004) * 400;
        const currentEnergyAccumulator = last.energySaved + (outputGain * (5 / 3600));

        const newPoint = {
          time: timeStr,
          ambientTemp: Number(ambient.toFixed(1)),
          basePanelTemp: Number(basePanel.toFixed(1)),
          cooledPanelTemp: Number(cooledPanel.toFixed(1)),
          powerOutputGain: Number(outputGain.toFixed(2)),
          energySaved: Number(currentEnergyAccumulator.toFixed(3))
        };
        
        setCurrentMetrics(newPoint);
        
        const newData = [...prevData, newPoint];
        if (newData.length > 20) newData.shift(); // Keep last 20
        return newData;
      });
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return { timeSeriesData: data, currentMetrics };
}
