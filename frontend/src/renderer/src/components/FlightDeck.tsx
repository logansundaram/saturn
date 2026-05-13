//shows the current agents that running as well as their status
//from the flight deck, the user can click into an agent which opens a tile in the TWM workspace

import ActiveAgent from "./ActiveAgent"

function FlightDeck(): React.JSX.Element {
    return (
        <div className="border-r-1 border-b-1 h-4/10 p-2">
            <h1>
                Flight Deck
            </h1>
            <div className="flex flex-col gap-2">
                <ActiveAgent/>    
                <ActiveAgent/>    
                <ActiveAgent/>    
            </div>
        </div>
    )
}

export default FlightDeck