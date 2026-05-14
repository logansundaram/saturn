import Sidebar from './components/Sidebar'
import Explorer from './components/Explorer'
import StatusBar from './components/StatusBar'
import Terminal from './components/Terminal'
import Workspace from './components/Workspace'

function App(): React.JSX.Element {
  return (
    <div className="flex h-screen w-screen">
      <div className="flex pb-6 flex-1">
        <Sidebar />
        <Explorer />
        <div className="flex flex-1 flex-col">
          <Workspace />
          <Terminal />
        </div>
      </div>
      <StatusBar />
    </div>
  )
}

export default App
