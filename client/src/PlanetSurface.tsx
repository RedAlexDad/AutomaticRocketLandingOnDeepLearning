import React from 'react';

interface PlanetSurfaceProps {
    width: number;
    height: number;
}

const PlanetSurface: React.FC<PlanetSurfaceProps> = ({ width, height }) => {
    return (
        <div style={{ width: `${width}px`, height: `${height}px`, background: 'lightblue' }}>
            {/* Ваш код для отображения поверхности планеты здесь */}
        </div>
    );
}

export default PlanetSurface;
