import SystemMonitor from './SystemMonitor'
import Logs from './Logs'
import FlightDeck from './FlightDeck'

function Explorer(): React.JSX.Element {
  return (
    <div>
      <FlightDeck />
      <SystemMonitor />
      <Logs />
    </div>
  )
}

export default Explorer
