//system monitor component, will ping the system useage/resources and siplay them
//currently thinking gpu, cpu ,vram, ram, but can add more relevant metrics
//should make the each resource modular
//realtime graphj owuld be cool too

import SystemResource from './SystemResource'

function SystemMonitor(): React.JSX.Element {
  return (
    <div className="p-2 border-r-1 border-b-1 h-3/10">
      <h1>System Monitor</h1>
      <SystemResource resource="GPU" max={100} useage={17.5} />
      <SystemResource resource="CPU" max={100} useage={10.5} />
      <SystemResource resource="RAM" max={96} useage={48} />
      <SystemResource resource="VRAM" max={24} useage={24} />
    </div>
  )
}

export default SystemMonitor
