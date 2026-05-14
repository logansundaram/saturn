import Sidebar from './components/Sidebar'
import Explorer from './components/Explorer'
import StatusBar from './components/StatusBar'

function App(): React.JSX.Element {
  return (
    <div className="flex h-screen w-screen">
      <div className="flex pb-6">
        <Sidebar/>
        <Explorer/>
      </div>
      <StatusBar/>
    </div>
  )   
}

export default App
