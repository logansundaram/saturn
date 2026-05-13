import { Bot, Split, Database, BrainCog, Wrench, Cog } from 'lucide-react';

function Sidebar(): React.JSX.Element {
  return (
    <div className="flex flex-col h-full justify-between border-r-1 border-black p-2">
        <div className="flex flex-col gap-2">
            <div className="sidebar-button">
                <Bot />
            </div>
            <div className="sidebar-button">
                <Split />
            </div>
            <div className="sidebar-button">
                <Database />
            </div>
            <div className="sidebar-button">
                <BrainCog />
            </div>
            <div className="sidebar-button">
                <Wrench />
            </div>
        </div>

        <div className="flex">
            <div className="sidebar-button">
                <Cog />
            </div>
        </div>
    </div>
  )
}


export default Sidebar