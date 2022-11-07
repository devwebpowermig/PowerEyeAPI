import { useState, useEffect } from 'react'

import styles from './LaserConfig.module.css';
export default function LaserConfig() {

    //position
    const [posX, setPosX] = useState(0)
    const [posY, setPosY] = useState(0)
    const [posZ, setPosZ] = useState(0)

    //rotation
    const [rotX, setRotX] = useState(0)
    const [rotY, setRotY] = useState(0)
    const [rotZ, setRotZ] = useState(0)

    //additional values
    const [area, setArea] = useState(0)
    const [width, setWidth] = useState(0)
    const [gap, setGap] = useState(0)
    const [divergence, setDivergence] = useState(0)

    //Job details
    const [ongoingJobId, setOngoingJobId] = useState(0)
    const [ongoingJobName, setOngoingJobName] = useState(0)

    useEffect(() => {
        returnValues()
    }, [rotX, rotY, rotZ, posX, posY, posZ, area, width, gap, divergence, ongoingJobId, ongoingJobName])

    function returnValues() {
        console.log(rotX, rotY, rotZ, posX, posY, posZ, area, width, gap, divergence, ongoingJobId, ongoingJobName)
    }

    return (
        <div>
            <h3>Posição Inicial</h3>
            <form id={styles.laserForm}>
                <label className={styles.laserInput}>Posição em X:
                    <input type="number" value={posX} onChange={(e) => setPosX(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Posição em Y:
                    <input type="number" value={posY} onChange={(e) => setPosY(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Posição em Z:
                    <input type="number" value={posZ} onChange={(e) => setPosZ(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Rotação em X:
                    <input type="number" value={rotX} onChange={(e) => setRotX(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Rotação em Y:
                    <input type="number" value={rotY} onChange={(e) => setRotY(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Rotação em Z:
                    <input type="number" value={rotZ} onChange={(e) => setRotZ(e.target.value)} />
                </label>
                <h3>Valores Adicionais</h3>
                <label className={styles.laserInput}>Área:
                    <input type="number" value={area} onChange={(e) => setArea(e.target.value)} />
                </label >
                <label className={styles.laserInput}>Largura:
                    <input type="number" value={width} onChange={(e) => setWidth(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Lacuna:
                    <input type="number" value={gap} onChange={(e) => setGap(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Divergence:
                    <input type="number" value={divergence} onChange={(e) => setDivergence(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Id da tarefa:
                    <input type="number" value={ongoingJobId} onChange={(e) => setOngoingJobId(e.target.value)} />
                </label>
                <label className={styles.laserInput}>Nome da tarefa:
                    <input type="number" value={ongoingJobName} onChange={(e) => setOngoingJobName(e.target.value)} />
                </label>

            </form>

        </div>
    )
}
