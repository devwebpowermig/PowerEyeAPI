import { IconContext } from "react-icons";
import { GiLaserburn } from "react-icons/gi";
import { TbRoute } from "react-icons/tb";
import { MdSensors } from "react-icons/md";
import LaserConfig from "../components/LaserConfig/LaserConfig";

const Laser = () => {
    return (
        <IconContext.Provider value={{ size: '1.5em' }}>
            <div>
                < GiLaserburn />
            </div>
        </IconContext.Provider>
    )

}
const Sensor = () => {
    return (
        <IconContext.Provider value={{ size: '1.5em' }}>
            <div>
                < MdSensors />
            </div>
        </IconContext.Provider>
    )

}
const Joint = () => {
    return (
        <IconContext.Provider value={{ size: '1.5em' }}>
            <div>
                < TbRoute />
            </div>
        </IconContext.Provider>
    )

}

export const navData = [
    {
        id: 0,
        icon: < Sensor />,
        text: "Sensor",
        component: "/"
    },
    {
        id: 1,
        icon: < Laser />,
        text: "Laser",
        component: <LaserConfig />
    },
    {
        id: 2,
        icon: < Joint />,
        text: "Joint",
        component: "joint"
    }

]