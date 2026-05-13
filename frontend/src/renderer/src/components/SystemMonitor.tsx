//system monitor component, will ping the system useage/resources and siplay them
//currently thinking gpu, cpu ,vram, ram, but can add more relevant metrics
//should make the each resource modular
//realtime graphj owuld be cool too


function SystemMonitor(): React.JSX.Element {
    return (
        <div className="p-2 border-r-1 h-6/10">
            <h1>
                System Monitor
            </h1>
            <p>
                GPU
            </p>
            <p>
                CPU
            </p>
            <p>
                VRAM
            </p>
            <p>
                RAM
            </p>
        </div>
    )
}

export default SystemMonitor