//todo: create a prop for log to display the agent run and relevant info in a concise way
//need interface and the update the main Logs.tsx file
//status could be optional, pass in fail or completed

interface LogProps {
  agentName: string
  dateRan: number
  status: string
}

function Log({ agentName, dateRan, status }: LogProps): React.JSX.Element {
  return (
    <div className="text-xs">
      <div>{agentName}</div>
      <div></div>
    </div>
  )
}

export default Log
