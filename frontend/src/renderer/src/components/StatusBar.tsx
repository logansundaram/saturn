function StatusBar(): React.JSX.Element {
    return (
        <div  className="fixed bottom-0 left-0 w-full border-t-1 h-6 flex items-center justify-between text-xs px-2">
            <div>
                Local Runtime | 4 Agents | GPU 71% | RAM 22GB
            </div>
            <div className="">
                GPT-5 | 34 tok/s | 1.2k tok | 230ms            
            </div>
        </div>
    )
}

export default StatusBar