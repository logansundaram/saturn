//modular component to be used in the SystemMonitor
//should have props for the resource name, current usage, and max capacity
interface SystemResourceProps {
  resource: string
  useage: number
  max: number
}

function SystemResource({ resource, useage, max }: SystemResourceProps): React.JSX.Element {
  return (
    <div className="text-xs">
      <div>
        {resource}: {useage}/{max}
      </div>
    </div>
  )
}

export default SystemResource
