import Sidebar from './components/Sidebar'
import Explorer from './components/Explorer'

function App(): React.JSX.Element {
  return (
    <div className="flex h-screen w-screen">
      <Sidebar/>
      <Explorer/>
    </div>
  )
}

export default App
