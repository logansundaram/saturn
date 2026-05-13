//active agent component for me to use in flight deck. should display 
//all the active agents as well update in real time, red for agent stopped, 
//yellow for awaiting approval for a tool call, blue for runnning, green for completed
//can also show metrics and other info, but tbd for now
function ActiveAgent(): React.JSX.Element {
    return (
        <div className="border-1 flex p-1 rounded">
            <h1 className="text-sm">
                Agent
            </h1>
        </div>
    )
}

export default ActiveAgent