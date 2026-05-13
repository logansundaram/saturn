import SystemMonitor from './SystemMonitor'
import RecentAgents from './RecentAgents'

function Explorer(): React.JSX.Element {
    return (
        <div>
            <RecentAgents />
            <SystemMonitor />
        </div>
    )
}

export default Explorer