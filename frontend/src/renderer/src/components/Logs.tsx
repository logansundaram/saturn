//need to create modular log component to lead previous agent runs
//could be useful to open up a logs panel in the tiling window manager
//maybe add a search?
import Log from './Log'

function Logs(): React.JSX.Element {
  return (
    <div className="h-3/10 border-r-1">
      <div>
        <h1>Logs</h1>
      </div>
      <div className="flex flex-col">
        <Log status="completed" agentName="News Agent" dateRan={10} />
        <Log status="completed" agentName="News Agent" dateRan={10} />
        <Log status="completed" agentName="News Agent" dateRan={10} />
      </div>
    </div>
  )
}

export default Logs
