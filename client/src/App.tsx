import './App.css'
import { BrowserRouter } from "react-router-dom";
import Rocket from "./Rocket.tsx";
import PlanetSurface from "./PlanetSurface.tsx";

function App() {
    return (
        <BrowserRouter basename="/AutomaticRocketLandingOnDeepLearning">
            <div className="App">
                {/* Передаем координаты для отображения ракеты */}
                <Rocket x={100} y={100} />
                {/* Передаем ширину и высоту для отображения поверхности планеты */}
                <PlanetSurface width={500} height={500} />
            </div>
        </BrowserRouter>
    )
}

export default App;
