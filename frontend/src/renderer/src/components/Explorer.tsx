import SystemMonitor from './SystemMonitor'
import RecentAgents from './RecentAgents'
import FlightDeck from './FlightDeck'

function Explorer(): React.JSX.Element {
    return (
        <div>
            <FlightDeck />
            <SystemMonitor />
        </div>
    )
}

export default Explorer