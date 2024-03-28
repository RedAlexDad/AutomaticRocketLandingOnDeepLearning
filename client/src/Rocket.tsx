import React from 'react';

interface RocketProps {
    x: number;
    y: number;
}

const Rocket: React.FC<RocketProps> = ({ x, y }) => {
    return (
        <div style={{ position: 'absolute', left: `${x}px`, top: `${y}px`, width: '50px', height: '100px', background: 'red' }}>
            {/* Ваш код для отображения ракеты здесь */}
        </div>
    );
}

export default Rocket;
